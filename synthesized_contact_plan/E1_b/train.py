#!/usr/bin/env python3
"""E1_b: DiLoCo-style local SGD and asynchronous ring gossip, four GPU ranks.

Drop-in replacement for E1_b/train.py. Relies on the existing, locally installed
E1_b/e1a_reference.py copied from E1_a_new with install.sh. All four ranks
validate their *own* model on the identical holdout set. Original metric CSV
columns are retained and per-worker/aggregate validation files are added.
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torchvision import models

try:
    import e1a_reference as ref
except ImportError as exc:
    raise SystemExit("Missing e1a_reference.py. Run ./install.sh in E1_b first.") from exc

MSG_FIELDS = [
    'rank','host','step','epoch','unix_time_s','elapsed_s','event','src_rank',
    'dst_rank','src_ip','dst_ip','payload_bytes_logical','duration_host_s',
    'local_version','received_version','status','method','measurement_scope'
]
WORKER_VAL_FIELDS = [
    'experiment','method','rank','epoch','step','unix_time_s','elapsed_s',
    'val_loss','val_accuracy','val_samples','val_s',
    'local_samples','global_samples_approx'
]
AGG_VAL_FIELDS = [
    'experiment','method','epoch','step','elapsed_s','unix_time_s',
    'mean_accuracy','min_accuracy','max_accuracy','std_accuracy',
    'mean_loss','min_loss','max_loss','std_loss','val_samples_per_worker',
    'world_size','mean_accuracy_target_reached','all_workers_target_reached'
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--method', choices=['diloco','gossip'], required=True)
    ap.add_argument('--dataset', choices=['cifar100','fake'], default='cifar100')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--min-lr', type=float, default=1e-4)
    ap.add_argument('--warmup-epochs', type=int, default=5)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--label-smoothing', type=float, default=0.)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--eval-every-epochs', type=int, default=1)
    ap.add_argument('--log-every-steps', type=int, default=10)
    ap.add_argument('--max-seconds', type=float, default=3600)
    ap.add_argument('--target-accuracy', type=float, default=.70)
    ap.add_argument('--val-size', type=int, default=5000)
    ap.add_argument('--validation-source', choices=['train-holdout','official-test'], default='train-holdout')
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--train-subset', type=int, default=0)
    ap.add_argument('--val-subset', type=int, default=0)
    ap.add_argument('--sample-period-s', type=float, default=1.)
    ap.add_argument('--gpu-idle-threshold', type=float, default=20.)
    ap.add_argument('--net-iface', default='br0')
    ap.add_argument('--local-steps', type=int, default=32)
    ap.add_argument('--outer-lr', type=float, default=1.)
    ap.add_argument('--outer-momentum', type=float, default=.9)
    ap.add_argument('--gossip-every', type=int, default=32)
    ap.add_argument('--gossip-alpha', type=float, default=.5)
    ap.add_argument('--cpu-smoke', action='store_true')
    a = ap.parse_args()
    if min(a.epochs,a.batch_size,a.log_every_steps,a.eval_every_epochs,a.local_steps,a.gossip_every) < 1:
        ap.error('Epochs, batch size and intervals must be positive')
    if not 0 <= a.warmup_epochs < a.epochs:
        ap.error('--warmup-epochs must be in [0, epochs)')
    if not 0 < a.gossip_alpha <= 1 or not 0 <= a.outer_momentum < 1 or a.outer_lr <= 0:
        ap.error('Invalid gossip or outer optimizer parameters')
    return a


def open_csv(path, fields):
    fh = path.open('w', newline='', buffering=1)
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    return fh, writer


class AsyncRing:
    """Blocking Gloo send/recv in two background threads, bounded queues.

    All sends travel rank -> (rank+1)%world; receives from (rank-1)%world.
    The CPU training thread never blocks on a peer send during a minibatch.
    Thread shutdown uses an explicit in-band negative version marker.
    """
    def __init__(self, dist_group, num_params, rank, world):
        self.group, self.n = dist_group, num_params
        self.left, self.right = (rank-1)%world, (rank+1)%world
        self.outbox = queue.Queue(maxsize=1)
        self.inbox = queue.Queue(maxsize=2)
        self.events = queue.Queue()
        self.sender = threading.Thread(target=self.send_loop, daemon=True)
        self.receiver = threading.Thread(target=self.recv_loop, daemon=True)
        self.sender.start()
        self.receiver.start()

    def send_loop(self):
        while True:
            item = self.outbox.get()
            if item is None:
                try:
                    dist.send(torch.full((self.n+1,), -1., dtype=torch.float32),
                              dst=self.right, group=self.group)
                except Exception as exc:
                    self.events.put(('error',0,0,0.,str(exc)))
                return
            version, buffer, begin = item
            try:
                dist.send(buffer, dst=self.right, group=self.group)
                self.events.put(('send_complete',version,buffer.numel()*buffer.element_size(),
                                 time.perf_counter()-begin,'ok'))
            except Exception as exc:
                self.events.put(('error',version,0,time.perf_counter()-begin,str(exc)))
                return

    def recv_loop(self):
        buffer = torch.empty(self.n+1, dtype=torch.float32)
        try:
            while True:
                dist.recv(buffer, src=self.left, group=self.group)
                version = int(buffer[0].item())
                if version < 0:
                    return
                item = (version, buffer[1:].clone())
                try:
                    self.inbox.put_nowait(item)
                except queue.Full:
                    self.inbox.get_nowait()  # bounded memory, discard oldest stale snapshot
                    self.events.put(('inbox_drop',version,0,0.,'stale_snapshot_dropped'))
                    self.inbox.put_nowait(item)
        except Exception as exc:
            self.events.put(('error',0,0,0.,str(exc)))

    def submit(self, step, payload):
        try:
            self.outbox.put_nowait((step,payload,time.perf_counter()))
            return True
        except queue.Full:
            return False

    def shutdown(self, timeout=90):
        self.outbox.put(None)
        self.sender.join(timeout)
        self.receiver.join(timeout)
        return not (self.sender.is_alive() or self.receiver.is_alive())


def main():
    a = parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world < 2:
        raise RuntimeError('E1_b requires at least two ranks')
    if not a.cpu_smoke:
        torch.cuda.set_device(local_rank)
    device = torch.device('cpu' if a.cpu_smoke else f'cuda:{local_rank}')
    backend = 'gloo' if a.cpu_smoke else 'nccl'
    dist.init_process_group(backend=backend,init_method='env://',timeout=timedelta(minutes=30))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark=True
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ref.data_loaders is from E1_a_new and normally builds validation only on rank 0.
    train_loader, val_loader, sampler, train_n, val_n = ref.data_loaders(a,rank,world)
    if rank != 0:
        # Build the exact same evaluation DataLoader as rank 0, without changing
        # the local DistributedSampler or the worker's training data allocation.
        _, val_loader, _, _, local_val_n = ref.data_loaders(a,0,world)
        if local_val_n != val_n:
            raise RuntimeError('Validation split mismatch across ranks')
    if val_loader is None:
        raise RuntimeError('Could not construct validation loader on rank '+str(rank))

    model = models.resnet18(weights=None,num_classes=100)
    model.conv1 = nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    model.to(device)
    params=list(model.parameters())
    num_params=sum(p.numel() for p in params)
    for p in params:
        dist.broadcast(p.data,src=0)
    optimizer=torch.optim.SGD(params,lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay)
    criterion=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    amp=device.type=='cuda' and not a.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    random.seed(a.seed+rank)
    np.random.seed(a.seed+rank)
    host=socket.gethostname()
    ip=os.environ.get('E1_WORKER_IP',socket.gethostbyname(host))
    workers=[None]*world
    dist.all_gather_object(workers,{'rank':rank,'local_rank':local_rank,
                                     'node':host,'ip':ip,'pid':os.getpid()})
    peer_map={int(p['rank']):p for p in workers}
    if rank==0:
        (out/'worker_map.json').write_text(json.dumps({'workers':workers},indent=2))

    step_f, step_w = open_csv(out/f'steps_rank{rank}.csv',ref.STEP_FIELDS)
    comm_f, comm_w = open_csv(out/f'collectives_rank{rank}.csv',ref.COLL_FIELDS)
    sync_f, sync_w = open_csv(out/f'sync_events_rank{rank}.csv',ref.SYNC_FIELDS)
    msg_f, msg_w = open_csv(out/f'messages_rank{rank}.csv',MSG_FIELDS)
    wval_f, wval_w = open_csv(out/f'validation_rank{rank}.csv',WORKER_VAL_FIELDS)
    val_f = val_w = agg_f = agg_w = None
    if rank==0:
        val_f,val_w=open_csv(out/'validation.csv',ref.VAL_FIELDS)
        agg_f,agg_w=open_csv(out/'validation_aggregate.csv',AGG_VAL_FIELDS)

    anchor = velocity = None
    def sync_gpu():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def flattened():
        return torch.cat([p.data.detach().reshape(-1).float() for p in params])
    @torch.no_grad()
    def assign(vector):
        offset=0
        for p in params:
            count=p.numel()
            p.copy_(vector[offset:offset+count].view_as(p).to(dtype=p.dtype))
            offset+=count
    if a.method=='diloco':
        anchor=flattened().clone()
        velocity=torch.zeros_like(anchor)
    ring=None
    if a.method=='gossip':
        # Separate Gloo process group for CPU snapshot traffic (also on GPU hosts).
        ring=AsyncRing(dist.new_group(ranks=list(range(world)),backend='gloo'),num_params,rank,world)
    left=(rank-1)%world
    right=(rank+1)%world
    outer_round=0
    completed_sends=0
    received_messages=0
    comm_events=comm_bytes=comm_fail=0
    comm_sum=comm_max=comm_last=0.
    comm_first=float('inf')
    training_start=time.perf_counter()

    def elapsed():return time.perf_counter()-training_start
    def event(step,epoch,kind,src,dst,payload=0,duration=0.,version='',received='',status='ok'):
        nonlocal comm_events,comm_bytes,comm_fail,comm_sum,comm_max,comm_first,comm_last
        now=time.time()
        msg_w.writerow(dict(rank=rank,host=host,step=step,epoch=epoch,
            unix_time_s=round(now,6),elapsed_s=round(elapsed(),6),event=kind,
            src_rank=src,dst_rank=dst,
            src_ip=peer_map.get(src,{}).get('ip','') if isinstance(src,int) else '',
            dst_ip=peer_map.get(dst,{}).get('ip','') if isinstance(dst,int) else '',
            payload_bytes_logical=payload,duration_host_s=round(duration,7),
            local_version=version,received_version=received,status=status,
            method=a.method,measurement_scope='logical tensor payload; NOT physical TCP bytes'))
        if kind in ('outer_allreduce','send_complete'):
            comm_w.writerow(dict(rank=rank,step=step,bucket_index=0,bytes_logical=payload,
                start_unix_s=round(now-duration,6),end_unix_s=round(now,6),
                duration_host_s=round(duration,7),status=status,src_node=host,src_ip=ip,
                collective_membership=('diloco_delta_allreduce' if kind=='outer_allreduce'
                                       else 'gossip_p2p_send'),
                destination_rank=dst,
                destination_ip=peer_map.get(dst,{}).get('ip','') if isinstance(dst,int) else 'not_attributable',
                measurement_scope='logical tensor payload; not physical bytes'))
            comm_events+=1
            comm_bytes+=payload
            comm_sum+=duration
            comm_max=max(comm_max,duration)
            comm_last=time.perf_counter()
            comm_first=min(comm_first,comm_last-duration)
        elif kind=='error':comm_fail+=1

    def reset_comm():
        nonlocal comm_events,comm_bytes,comm_fail,comm_sum,comm_max,comm_first,comm_last
        comm_events=comm_bytes=comm_fail=0
        comm_sum=comm_max=comm_last=0.
        comm_first=float('inf')

    def communicate(step,epoch,force=False,poll_only=False):
        nonlocal anchor,velocity,outer_round,completed_sends,received_messages
        if a.method=='diloco':
            if not force and step%a.local_steps:return
            sync_gpu()
            t0=time.perf_counter()
            delta=anchor-flattened()
            dist.all_reduce(delta,op=dist.ReduceOp.SUM)
            delta.div_(world)
            velocity.mul_(a.outer_momentum).add_(delta)
            anchor.add_(velocity,alpha=-a.outer_lr)
            assign(anchor)
            sync_gpu()
            outer_round+=1
            event(step,epoch,'outer_allreduce',rank,'all',delta.numel()*delta.element_size(),
                  time.perf_counter()-t0,version=step)
            return
        # Nonblocking training-thread processing of completed network messages.
        while True:
            try:kind,version,payload,duration,status=ring.events.get_nowait()
            except queue.Empty:break
            if kind=='send_complete':
                event(step,epoch,kind,rank,right,payload,duration,version=version)
                completed_sends+=1
            else:event(step,epoch,kind,rank,right,0,duration,version=version,status=status)
        while True:
            try:version,snapshot=ring.inbox.get_nowait()
            except queue.Empty:break
            t0=time.perf_counter()
            incoming=snapshot.to(device)
            assign(flattened().mul_(1-a.gossip_alpha).add_(incoming,alpha=a.gossip_alpha))
            sync_gpu()
            event(step,epoch,'receive_mix',left,rank,snapshot.numel()*snapshot.element_size(),
                  time.perf_counter()-t0,version=step,received=version)
            received_messages+=1
        if not poll_only and step%a.gossip_every==0:
            t0=time.perf_counter()
            payload=torch.cat((torch.tensor([float(step)]),flattened().cpu()))
            if ring.submit(step,payload):
                event(step,epoch,'send_posted',rank,right,payload.numel()*payload.element_size(),
                      time.perf_counter()-t0,version=step,status='posted')
            else:event(step,epoch,'send_skipped_busy',rank,right,0,
                       time.perf_counter()-t0,version=step,status='busy')

    def sync_event(epoch,step,kind,t0,note=''):
        sync_w.writerow(dict(rank=rank,host=host,epoch=epoch,step=step,event=kind,
           unix_time_s=round(time.time(),6),duration_s=round(time.perf_counter()-t0,7),
           world_size=world,note=note))

    meta=dict(experiment='E1_b',method=a.method,world_size=world,
        model='ResNet18 CIFAR stem',dataset=a.dataset,train_n=train_n,val_n=val_n,
        global_batch_size=world*a.batch_size,parameters=num_params,
        arguments=vars(a),reference='E1_a',
        validation='Identical full holdout set independently evaluated on every rank; aggregate is mean of LOCAL accuracies, not a global model',
        communication='DiLoCo-style pseudo-gradient allreduce or async directed Gloo ring',
        tcp_note='Logical event bytes are not observed TCP bytes; original E1_a_new socket monitor reused',
        start_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__)
    if rank==0:(out/'manifest.json').write_text(json.dumps(meta,indent=2))

    # Original E1_a_new resource monitor and rootless TCP telemetry.
    tcp_proc=tcp_log=tcp_stop=None
    if local_rank==0 and os.environ.get('E1_TCP_MEASURE','0')=='1':
        short=host.split('.')[0]
        tcp_stop=out/f'monitor_stop_{short}'
        try:
            tcp_log=(out/f'monitor_launch_{short}.log').open('w',buffering=1)
            env=os.environ.copy();env['NET_IFACE']=a.net_iface
            tcp_proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('socket_counter_watch.py')),
                str(out),os.environ.get('E1_TCP_PERIOD','0.1'),os.environ.get('E1_TCP_DURATION','3600'),
                '--stop-file',str(tcp_stop)],stdout=tcp_log,stderr=subprocess.STDOUT,env=env)
            (out/f'monitor_parent_{short}.json').write_text(json.dumps({
                'host':short,'trainer_pid':os.getpid(),'monitor_pid':tcp_proc.pid,
                'started_unix_s':time.time()},indent=2))
        except Exception as exc:
            print('WARNING: TCP monitoring not started:',exc,flush=True)
    resource_monitor=None
    if device.type=='cuda' and a.sample_period_s>0:
        resource_monitor=ref.SamplerThread(out,rank,local_rank,a.net_iface,
                                           a.sample_period_s,time.time())
        resource_monitor.start()

    best=-1.;first_target=None;first_all_target=None;last_val=None
    step=0;samples=0;completed_epoch=0;reason='epochs_completed'
    try:
        t0=time.perf_counter();dist.barrier();sync_event(0,0,'initial_barrier',t0)
        training_start=time.perf_counter()
        for epoch in range(a.epochs):
            sampler.set_epoch(epoch)
            lr=ref.lr_at(epoch,a)
            for group in optimizer.param_groups:group['lr']=lr
            model.train()
            iterator=iter(train_loader)
            while True:
                wait_start=time.perf_counter()
                try:images,targets=next(iterator)
                except StopIteration:break
                t0=time.perf_counter();batch_wait=t0-wait_start
                images=images.to(device,non_blocking=(device.type=='cuda'))
                targets=targets.to(device,non_blocking=(device.type=='cuda'))
                sync_gpu();t1=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    preds=model(images);loss=criterion(preds,targets)
                sync_gpu();t2=time.perf_counter()
                scaler.scale(loss).backward();sync_gpu();t3=time.perf_counter()
                scaler.step(optimizer);scaler.update();sync_gpu();t4=time.perf_counter()
                communicate(step+1,epoch+1)
                sync_gpu();t5=time.perf_counter()
                step+=1;bs=targets.shape[0];samples+=bs
                if step%a.log_every_steps==0:
                    step_w.writerow(dict(experiment='E1_b',host=host,rank=rank,local_rank=local_rank,
                      epoch=epoch+1,step=step,unix_time_s=round(time.time(),6),
                      elapsed_s=round(elapsed(),6),loss=round(loss.item(),7),lr=round(lr,9),
                      local_samples=samples,global_samples_approx=world*samples,batch_samples=bs,
                      step_s=round(t5-t0,7),batch_wait_s=round(batch_wait,7),h2d_s=round(t1-t0,7),
                      forward_s=round(t2-t1,7),backward_with_comm_s=round(t3-t2,7),
                      optimizer_s=round(t4-t3,7),other_s=round(t5-t4,7),
                      collectives=comm_events,collective_bytes_logical=comm_bytes,
                      collective_sum_host_s=round(comm_sum,7),
                      collective_span_host_s=round(max(0,comm_last-comm_first),7) if comm_events else 0.,
                      collective_max_host_s=round(comm_max,7),collective_failures=comm_fail,
                      gpu_memory_allocated_mb=round(torch.cuda.memory_allocated(device)/1048576,2) if device.type=='cuda' else 0.,
                      samples_per_s=round(bs/max(t5-t0,1e-9),3),step_wall_s=round(t5-wait_start,7)))
                    reset_comm()
                if rank==0 and step%100==0:
                    print(f'E1_b {a.method} epoch={epoch+1} step={step} elapsed={elapsed():.1f}s',flush=True)
            if a.method=='diloco' and step%a.local_steps:
                communicate(step,epoch+1,force=True)
            if a.method=='gossip':
                communicate(step,epoch+1,poll_only=True)
            completed_epoch=epoch+1
            t0=time.perf_counter();dist.barrier()
            sync_event(completed_epoch,step,'pre_validation_barrier',t0,
                       'epoch-boundary measurement synchronization; no model averaging')

            if completed_epoch%a.eval_every_epochs==0 or completed_epoch==a.epochs:
                model.eval()
                sync_gpu();t_val=time.perf_counter()
                correct=0;sum_loss=0.;num=0
                with torch.inference_mode():
                    for imgs,labels in val_loader:
                        imgs=imgs.to(device,non_blocking=(device.type=='cuda'))
                        labels=labels.to(device,non_blocking=(device.type=='cuda'))
                        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                            outputs=model(imgs);lv=criterion(outputs,labels)
                        count=labels.shape[0]
                        correct+=int((outputs.argmax(1)==labels).sum().item())
                        sum_loss+=float(lv.item())*count
                        num+=count
                sync_gpu();val_duration=time.perf_counter()-t_val
                local_accuracy=correct/max(num,1)
                local_loss=sum_loss/max(num,1)
                stamp=elapsed()
                wval_w.writerow(dict(experiment='E1_b',method=a.method,rank=rank,
                    epoch=completed_epoch,step=step,unix_time_s=round(time.time(),6),
                    elapsed_s=round(stamp,6),val_loss=round(local_loss,7),
                    val_accuracy=round(local_accuracy,7),val_samples=num,
                    val_s=round(val_duration,6),local_samples=samples,
                    global_samples_approx=world*samples))
                # All ranks evaluate independently on identical samples, then send only
                # small scalars to rank 0. No optimizer/model state is synchronized.
                local_scores=torch.tensor([local_accuracy,local_loss,float(num)],
                                          dtype=torch.float64,device=device)
                gathered=[torch.zeros_like(local_scores) for _ in range(world)]
                dist.all_gather(gathered,local_scores)
                # Per-worker checkpoint: preserves the actual divergent model.
                torch.save(dict(rank=rank,method=a.method,model=model.state_dict(),
                   optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                   epoch_completed=completed_epoch,step=step,elapsed_s=stamp,
                   config=vars(a),world_size=world,
                   outer_round=outer_round,anchor=anchor,outer_velocity=velocity),
                   out/f'checkpoint_rank{rank}.pt')
                if rank==0:
                    scores=torch.stack(gathered).cpu().numpy()
                    accuracies=scores[:,0];losses=scores[:,1]
                    if not np.all(scores[:,2]==scores[0,2]):
                        raise RuntimeError('Validation sample counts differ across ranks')
                    avg_acc=float(np.mean(accuracies))
                    min_acc=float(np.min(accuracies))
                    max_acc=float(np.max(accuracies))
                    std_acc=float(np.std(accuracies))
                    mean_target=int(a.target_accuracy>0 and avg_acc>=a.target_accuracy)
                    all_target=int(a.target_accuracy>0 and min_acc>=a.target_accuracy)
                    if first_target is None and mean_target:first_target=stamp
                    if first_all_target is None and all_target:first_all_target=stamp
                    agg_w.writerow(dict(experiment='E1_b',method=a.method,
                      epoch=completed_epoch,step=step,elapsed_s=round(stamp,6),
                      unix_time_s=round(time.time(),6),mean_accuracy=round(avg_acc,7),
                      min_accuracy=round(min_acc,7),max_accuracy=round(max_acc,7),
                      std_accuracy=round(std_acc,7),mean_loss=round(float(np.mean(losses)),7),
                      min_loss=round(float(np.min(losses)),7),max_loss=round(float(np.max(losses)),7),
                      std_loss=round(float(np.std(losses)),7),val_samples_per_worker=num,
                      world_size=world,mean_accuracy_target_reached=mean_target,
                      all_workers_target_reached=all_target))
                    # Keep the legacy E1_a_new validation.csv schema; explicitly rank0.
                    best=max(best,local_accuracy)
                    last_val=dict(experiment='E1_b',epoch=completed_epoch,step=step,
                      global_samples_approx=world*samples,unix_time_s=round(time.time(),6),
                      elapsed_s=round(stamp,6),val_loss=round(local_loss,7),
                      val_accuracy=round(local_accuracy,7),val_samples=num,
                      val_s=round(val_duration,6),lr=round(lr,9),
                      best_val_accuracy=round(best,7),
                      cumulative_samples_per_s=round(world*samples/max(stamp,1e-9),3))
                    val_w.writerow(last_val)
                    print('VALIDATION_ALL',json.dumps({'epoch':completed_epoch,
                        'mean_accuracy':avg_acc,'min_accuracy':min_acc,
                        'max_accuracy':max_acc,'std_accuracy':std_acc}),flush=True)
                    # Backward compatible checkpoint = rank 0 only.
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),epoch_completed=completed_epoch,
                        step=step,elapsed_s=stamp,config=vars(a),world_size=world,
                        best_val_accuracy=best,method=a.method,outer_round=outer_round,
                        anchor=anchor,outer_velocity=velocity),out/'checkpoint_latest.pt')
                model.train()

            t0=time.perf_counter();dist.barrier();sync_event(completed_epoch,step,'post_validation_barrier',t0)
            # Rank 0 alone decides termination, with epoch-boundary semantics.
            stop=torch.tensor([int(a.max_seconds>0 and elapsed()>=a.max_seconds)],
                              dtype=torch.int,device=device)
            t0=time.perf_counter();dist.broadcast(stop,src=0)
            sync_event(completed_epoch,step,'stop_decision_broadcast',t0)
            if stop.item():reason='time_budget';break

        if a.method=='gossip':
            ok=ring.shutdown()
            communicate(step,completed_epoch,poll_only=True)
            if not ok:
                event(step,completed_epoch,'background_timeout',rank,right,status='timeout')
        if rank==0:
            summary=dict(experiment='E1_b',method=a.method,status='ok',reason=reason,
              world_size=world,completed_epochs=completed_epoch,steps_per_worker=step,
              global_samples_approx=world*samples,wall_training_s=round(elapsed(),4),
              best_validation_accuracy_rank0=round(best,7),
              mean_accuracy_target_first_s=first_target,
              all_workers_target_first_s=first_all_target,
              target_accuracy=a.target_accuracy,latest_validation_rank0=last_val,
              validation_definition='Mean of per-rank local-model accuracies; not a global model')
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('SUMMARY',json.dumps(summary),flush=True)
        t0=time.perf_counter();dist.barrier();sync_event(completed_epoch,step,'final_barrier',t0)
    finally:
        if tcp_proc is not None:
            try:
                tcp_stop.touch()
                try:tcp_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    tcp_proc.terminate();tcp_proc.wait(timeout=3)
            except Exception as exc:print('WARNING: TCP monitor shutdown:',exc,flush=True)
        if tcp_log is not None:tcp_log.close()
        if resource_monitor is not None:resource_monitor.close()
        for handle in (step_f,comm_f,sync_f,msg_f,wval_f,val_f,agg_f):
            if handle is not None:handle.close()
        try:dist.destroy_process_group()
        except Exception as exc:print('WARNING: distributed shutdown:',exc,flush=True)

if __name__=='__main__':main()
