#!/usr/bin/env python3
"""E1_a: instrumented, unchanged synchronous DDP algorithm (4 GPU ranks).
Research measurement version of synthesized_contact_plan/E1_4workers.
"""
import argparse, csv, json, math, os, random, socket, subprocess, threading, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms

STEP_FIELDS=['experiment','host','rank','local_rank','epoch','step','unix_time_s','elapsed_s','loss','lr','local_samples','global_samples_approx','batch_samples','step_s','batch_wait_s','h2d_s','forward_s','backward_with_comm_s','optimizer_s','other_s','collectives','collective_bytes_logical','collective_sum_host_s','collective_span_host_s','collective_max_host_s','collective_failures','gpu_memory_allocated_mb','samples_per_s','step_wall_s']
VAL_FIELDS=['experiment','epoch','step','global_samples_approx','unix_time_s','elapsed_s','val_loss','val_accuracy','val_samples','val_s','lr','best_val_accuracy','cumulative_samples_per_s']
COLL_FIELDS=['rank','step','bucket_index','bytes_logical','start_unix_s','end_unix_s','duration_host_s','status','src_node','src_ip','collective_membership','destination_rank','destination_ip','measurement_scope']
SYNC_FIELDS=['rank','host','epoch','step','event','unix_time_s','duration_s','world_size','note']


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='../data')
    p.add_argument('--dataset',choices=['cifar100','fake'],default='cifar100')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32,help='per rank; global batch=4*batch-size')
    p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--min-lr',type=float,default=.0001)
    p.add_argument('--warmup-epochs',type=int,default=5)
    p.add_argument('--momentum',type=float,default=.9)
    p.add_argument('--weight-decay',type=float,default=.0005)
    p.add_argument('--label-smoothing',type=float,default=0.)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--eval-every-epochs',type=int,default=1)
    p.add_argument('--log-every-steps',type=int,default=10)
    p.add_argument('--max-seconds',type=float,default=3600)
    p.add_argument('--target-accuracy',type=float,default=.70)
    p.add_argument('--val-size',type=int,default=5000)
    p.add_argument('--validation-source',choices=['train-holdout','official-test'],default='train-holdout')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--train-subset',type=int,default=0)
    p.add_argument('--val-subset',type=int,default=0)
    p.add_argument('--sample-period-s',type=float,default=1.)
    p.add_argument('--gpu-idle-threshold',type=float,default=20.)
    p.add_argument('--net-iface',default='br0')
    p.add_argument('--collective-trace-every',type=int,default=0,help='0: aggregate only; N: per-bucket records on steps divisible by N')
    p.add_argument('--no-collective-hook',action='store_true',help='diagnostic; retains DDP default reduction but cannot count collectives')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.eval_every_epochs,a.log_every_steps)<1:p.error('counts must be positive')
    if not 0<=a.warmup_epochs<a.epochs:p.error('invalid warmup epochs')
    return a


def lr_at(epoch,a):
    if a.warmup_epochs and epoch<a.warmup_epochs:return a.lr*(epoch+1)/a.warmup_epochs
    frac=(epoch-a.warmup_epochs)/max(1,a.epochs-a.warmup_epochs-1)
    return a.min_lr+.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*min(1,max(0,frac))))


def data_loaders(a,rank,world):
    mean=(.5071,.4867,.4408);std=(.2675,.2565,.2761)
    t_train=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
    t_val=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    if a.dataset=='fake':
        train=datasets.FakeData(size=512,image_size=(3,32,32),num_classes=100,transform=t_train)
        val=datasets.FakeData(size=128,image_size=(3,32,32),num_classes=100,transform=t_val,random_offset=50000)
    else:
        original=datasets.CIFAR100(a.data_root,train=True,download=False,transform=t_train)
        evaluation=datasets.CIFAR100(a.data_root,train=True,download=False,transform=t_val)
        if a.validation_source=='train-holdout':
            labels=np.asarray(original.targets); rng=np.random.default_rng(a.seed)
            buckets=[rng.permutation(np.where(labels==c)[0]) for c in range(100)]
            q,r=divmod(a.val_size,100)
            vi=np.concatenate([v[:q+(c<r)] for c,v in enumerate(buckets)]).tolist()
            ti=np.concatenate([v[q+(c<r):] for c,v in enumerate(buckets)]).tolist()
            train=Subset(original,ti);val=Subset(evaluation,vi)
        else:
            train=original;val=datasets.CIFAR100(a.data_root,train=False,download=False,transform=t_val)
    if a.train_subset:train=Subset(train,range(min(a.train_subset,len(train))))
    if a.val_subset:val=Subset(val,range(min(a.val_subset,len(val))))
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=False)
    kwargs=dict(num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    tr=DataLoader(train,batch_size=a.batch_size,sampler=sampler,**kwargs)
    va=DataLoader(val,batch_size=max(128,a.batch_size),shuffle=False,**kwargs) if rank==0 else None
    return tr,va,sampler,len(train),len(val)


class CollectiveMeter:
    """DDP default dense all-reduce, via a counting/timing comm hook.

    Durations are HOST elapsed time between launching a NCCL async operation
    and its Future callback completion, NOT isolated GPU kernel duration.
    Multiple bucket operations may overlap; sums are NOT additively critical time.
    """
    def __init__(self,world,rank,trace_writer,trace_every,peer_meta=None):
        self.world=world;self.rank=rank;self.writer=trace_writer;self.trace_every=trace_every
        self.peer_meta = peer_meta if peer_meta is not None else {}
        self.lock=threading.Lock();self.step=0;self.entries=[];self.errors=[]

    def begin(self,step):
        with self.lock:self.step=step;self.entries=[];self.errors=[]

    def hook(self,state,bucket):
        tensor=bucket.buffer();size=tensor.numel()*tensor.element_size()
        with self.lock:step=self.step
        start_perf=time.perf_counter();start_unix=time.time()
        try:
            fut=dist.all_reduce(tensor,op=dist.ReduceOp.SUM,async_op=True).get_future()
        except Exception as e:
            with self.lock:self.errors.append(f'step {step}: {e}')
            raise
        def finish(done):
            try:
                result=done.value()[0]
                result.div_(self.world)
                end_perf=time.perf_counter();end_unix=time.time()
                with self.lock:
                    self.entries.append((start_perf,end_perf,size))
                    if self.writer is not None and self.trace_every>0 and step%self.trace_every==0:
                        self.writer.writerow(dict(rank=self.rank,step=step,bucket_index=bucket.index(),
                            bytes_logical=size,start_unix_s=round(start_unix,6),end_unix_s=round(end_unix,6),
                            duration_host_s=round(end_perf-start_perf,7),status='ok',
                            src_node=socket.gethostname(),src_ip=self.peer_meta.get(self.rank,{}).get('ip',''),
                            collective_membership='all_ranks',destination_rank='not_attributable',destination_ip='not_attributable',
                            measurement_scope='DDP bucket logical allreduce; NOT per-peer wire transfer'))
                return result
            except Exception as e:
                with self.lock:self.errors.append(f'step {step}: {e}')
                raise
        return fut.then(finish)

    def summary(self):
        with self.lock:entries=list(self.entries);fail=len(self.errors)
        return dict(collectives=len(entries),collective_bytes_logical=sum(x[2] for x in entries),
                    collective_sum_host_s=sum(x[1]-x[0] for x in entries),
                    collective_span_host_s=max(x[1] for x in entries)-min(x[0] for x in entries) if entries else 0.,
                    collective_max_host_s=max((x[1]-x[0] for x in entries),default=0.),
                    collective_failures=fail)


class SamplerThread:
    def __init__(self,output,rank,local_rank,iface,period,start_clock):
        self.stop=threading.Event();self.path=output/f'resources_rank{rank}.csv'
        self.rank=rank;self.local=local_rank;self.iface=iface;self.period=period;self.start_clock=start_clock
        self.thread=threading.Thread(target=self.run,name=f'monitor-rank{rank}',daemon=True)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(timeout=max(5.,2*self.period))
    def run(self):
        fields=['rank','local_rank','host','unix_time_s','elapsed_s','gpu_util_pct','gpu_memory_util_pct','gpu_mem_used_mib','gpu_power_w','rx_bytes_node','tx_bytes_node','nic','status']
        with self.path.open('w',newline='',buffering=1) as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            while not self.stop.is_set():
                now=time.time(); values=['','','',''];status='ok'
                try:
                    p=subprocess.run(['nvidia-smi','-i',str(self.local),
                       '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4,check=True)
                    values=[v.strip() for v in p.stdout.strip().splitlines()[0].split(',')]
                    if len(values)!=4:raise ValueError('unexpected nvidia-smi column count')
                except (subprocess.SubprocessError,OSError,IndexError,ValueError) as e:status='gpu_read_failed:'+type(e).__name__
                rx=tx=''
                if self.local==0: # node-level counters, once per node; NOT a per-process measure.
                    try:
                        root=Path('/sys/class/net')/self.iface/'statistics'
                        rx=int((root/'rx_bytes').read_text());tx=int((root/'tx_bytes').read_text())
                    except (OSError,ValueError):status+=';network_unavailable'
                w.writerow(dict(rank=self.rank,local_rank=self.local,host=socket.gethostname(),
                    unix_time_s=round(now,6),elapsed_s=round(now-self.start_clock,6),
                    gpu_util_pct=values[0],gpu_memory_util_pct=values[1],gpu_mem_used_mib=values[2],
                    gpu_power_w=values[3],rx_bytes_node=rx,tx_bytes_node=tx,nic=self.iface,status=status))
                self.stop.wait(self.period)


def run():
    a=cli();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',init_method='env://',timeout=timedelta(minutes=15))
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    train_loader,val_loader,sampler,train_n,val_n=data_loaders(a,rank,world)
    base=models.resnet18(weights=None,num_classes=100)
    base.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False);base.maxpool=nn.Identity()
    base.to(device)
    ddp=DDP(base,device_ids=[local],output_device=local)
    opt=torch.optim.SGD(ddp.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay,nesterov=False)
    crit=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    amp=not a.no_amp;scaler=torch.amp.GradScaler('cuda',enabled=amp)
    random.seed(a.seed+rank);np.random.seed(a.seed+rank)
    steps_f=(output/f'steps_rank{rank}.csv').open('w',newline='',buffering=1);sw=csv.DictWriter(steps_f,fieldnames=STEP_FIELDS);sw.writeheader()
    coll_f=(output/f'collectives_rank{rank}.csv').open('w',newline='',buffering=1)
    cw=csv.DictWriter(coll_f,fieldnames=COLL_FIELDS);cw.writeheader()
    # One-time metadata exchange BEFORE communication measurements start.
    ip=os.environ.get("E1_WORKER_IP") or socket.gethostbyname(socket.gethostname())
    peers=[None]*world
    dist.all_gather_object(peers,dict(rank=rank,local_rank=local,node=socket.gethostname(),ip=ip,pid=os.getpid()))
    peers={int(v['rank']):v for v in peers}
    if rank==0:(output/'worker_map.json').write_text(json.dumps({'workers':list(peers.values()),'note':'rank-to-IP mapping; IP is shared by ranks on the same physical node; NCCL internal peer flows not inferred'},indent=2))
    meter=CollectiveMeter(world,rank,cw,a.collective_trace_every,peers)
    sync_f=(output/f'sync_events_rank{rank}.csv').open('w',newline='',buffering=1)
    syncw=csv.DictWriter(sync_f,fieldnames=SYNC_FIELDS);syncw.writeheader()
    def sync_event(epoch,step,event,start_perf,note=''):
        syncw.writerow(dict(rank=rank,host=socket.gethostname(),epoch=epoch,step=step,event=event,
            unix_time_s=round(time.time(),6),duration_s=round(time.perf_counter()-start_perf,7),
            world_size=world,note=note))
    if not a.no_collective_hook:ddp.register_comm_hook(meter,meter.hook)
    vf=None
    if rank==0:
        vf=(output/'validation.csv').open('w',newline='',buffering=1);vw=csv.DictWriter(vf,fieldnames=VAL_FIELDS);vw.writeheader()
        meta=dict(experiment='E1_a',start_utc=datetime.now(timezone.utc).isoformat(),world_size=world,host=socket.gethostname(),
          gpu=torch.cuda.get_device_name(local),torch=torch.__version__,model='ResNet18 CIFAR stem',
          train_samples=train_n,val_samples=val_n,global_batch_size=world*a.batch_size,arguments=vars(a),
          baseline='synthesized_contact_plan/E1_4workers on GitHub as of 2026-11-06',
          semantics={'collective_bytes_logical':'sum of gradient tensor bytes submitted per bucket per rank, not physical network bytes',
              'collective_sum_host_s':'sum of async NCCL Future completion durations; overlapping operations double-count',
              'collective_span_host_s':'last callback minus first launch, host-side span, not GPU NCCL kernel duration',
              'backward_with_comm_s':'backpropagation incl overlapped AllReduce; not additive with collective time',
              'step_s':'training step duration after batch available, excludes dataloader next() wait',
              'batch_wait_s':'time spent in next(loader) including initial iterator and epoch overhead as seen by training process',
              'resources':'GPU samples per rank; NIC RX/TX counters node-global on local_rank==0 only, includes non-training traffic',
              'gpu_idle_proxy':'fraction of device samples below threshold; not precise GPU idle time',
              'contact_metrics':'not applicable to E1_a stable link; E2 contact trace required for utilization/outage analysis',
              'time_to_accuracy':'first observed validation crossing; sampled at epoch boundaries',
              'instrumentation':'registered custom DDP comm hook, same dense average as default DDP; timing overhead may change throughput',
              'peer_metadata':'worker_map.json lists ranks, hostnames, IPs; DDP collectives do not expose exact per-rank NCCL transport byte paths' })
        (output/'manifest.json').write_text(json.dumps(meta,indent=2,default=str))
    best=-1.;first_target=None;last_val=None;steps=0;samples=0;completed=0;reason='epochs_completed'
    tbar=time.perf_counter();dist.barrier();sync_event(0,0,'initial_barrier',tbar)
    clock=time.time();begin=time.perf_counter()
    # Start the rootless TCP monitor INSIDE the node's local-rank-0 training process.
    # This avoids a second remote oarsh launch, readiness races, and global stop files.
    tcp_monitor=None
    tcp_monitor_log=None
    tcp_stop=None
    if local==0 and os.environ.get('E1_TCP_MEASURE','0')=='1':
        host_short=socket.gethostname().split('.')[0]
        tcp_stop=output/f'monitor_stop_{host_short}'
        monitor_script=Path(__file__).resolve().with_name('socket_counter_watch.py')
        try:
            tcp_monitor_log=(output/f'monitor_launch_{host_short}.log').open('w',buffering=1)
            env=os.environ.copy();env['NET_IFACE']=a.net_iface
            tcp_monitor=subprocess.Popen([os.sys.executable,str(monitor_script),str(output),
                os.environ.get('E1_TCP_PERIOD','0.1'),os.environ.get('E1_TCP_DURATION','3600'),
                '--stop-file',str(tcp_stop)],stdout=tcp_monitor_log,stderr=subprocess.STDOUT,env=env)
            (output/f'monitor_parent_{host_short}.json').write_text(json.dumps({
                'host':host_short,'trainer_pid':os.getpid(),'monitor_pid':tcp_monitor.pid,
                'started_unix_s':time.time()},indent=2))
            print(f'Rootless TCP monitor launched host={host_short} pid={tcp_monitor.pid}',flush=True)
        except Exception as e:
            print(f'WARNING: rootless TCP monitor unavailable on {host_short}: {e}',flush=True)
            if tcp_monitor_log:tcp_monitor_log.close();tcp_monitor_log=None
            tcp_monitor=None
    monitoring=SamplerThread(output,rank,local,a.net_iface,a.sample_period_s,clock) if a.sample_period_s>0 else None
    if monitoring:monitoring.start()
    try:
        for epoch in range(a.epochs):
            sampler.set_epoch(epoch);lr=lr_at(epoch,a)
            for group in opt.param_groups:group['lr']=lr
            ddp.train();iterator=iter(train_loader)
            while True:
                t_wait=time.perf_counter()
                try:images,labels=next(iterator)
                except StopIteration:break
                t0=time.perf_counter();data_wait=t0-t_wait
                images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
                torch.cuda.synchronize();t1=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                meter.begin(steps+1)
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    logits=ddp(images);loss=crit(logits,labels)
                torch.cuda.synchronize();t2=time.perf_counter()
                scaler.scale(loss).backward()
                torch.cuda.synchronize();t3=time.perf_counter()
                scaler.step(opt);scaler.update();torch.cuda.synchronize();t4=time.perf_counter()
                steps+=1;bs=labels.size(0);samples+=bs
                stat=meter.summary() if not a.no_collective_hook else dict(collectives=0,collective_bytes_logical=0,collective_sum_host_s=0,collective_span_host_s=0,collective_max_host_s=0,collective_failures=0)
                if steps%a.log_every_steps==0:
                    sw.writerow(dict(experiment='E1_a',host=socket.gethostname(),rank=rank,local_rank=local,epoch=epoch+1,step=steps,
                        unix_time_s=round(time.time(),6),elapsed_s=round(t4-begin,6),loss=round(float(loss.item()),7),lr=round(lr,9),
                        local_samples=samples,global_samples_approx=world*samples,batch_samples=bs,
                        step_s=round(t4-t0,7),batch_wait_s=round(data_wait,7),h2d_s=round(t1-t0,7),
                        forward_s=round(t2-t1,7),backward_with_comm_s=round(t3-t2,7),optimizer_s=round(t4-t3,7),other_s=0,
                        collectives=stat['collectives'],collective_bytes_logical=stat['collective_bytes_logical'],
                        collective_sum_host_s=round(stat['collective_sum_host_s'],7),collective_span_host_s=round(stat['collective_span_host_s'],7),
                        collective_max_host_s=round(stat['collective_max_host_s'],7),collective_failures=stat['collective_failures'],
                        gpu_memory_allocated_mb=round(torch.cuda.memory_allocated(device)/1048576,2),
                        samples_per_s=round(bs/max(t4-t0,1e-9),3),step_wall_s=round(t4-t_wait,7)))
                if rank==0 and steps%100==0:
                    print(f'E1_a epoch={epoch+1} step={steps} elapsed={t4-begin:.1f}s loss={loss.item():.4f} collectives={stat["collectives"]}',flush=True)
            completed=epoch+1
            epoch_barrier_start=time.perf_counter();dist.barrier()
            sync_event(completed,steps,'pre_validation_barrier',epoch_barrier_start,'ranks 1..N wait until all completed their epoch')
            if rank==0 and (completed%a.eval_every_epochs==0 or completed==a.epochs):
                base.eval();torch.cuda.synchronize();ev0=time.perf_counter();correct=0;loss_total=0.;n=0
                with torch.inference_mode():
                    for images,labels in val_loader:
                        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
                        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                            logits=base(images);loss_val=crit(logits,labels)
                        b=labels.size(0);n+=b;correct+=int((logits.argmax(1)==labels).sum().item());loss_total+=float(loss_val.item())*b
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin;accuracy=correct/n;best=max(best,accuracy)
                if first_target is None and a.target_accuracy>0 and accuracy>=a.target_accuracy:first_target=elapsed
                last_val=dict(experiment='E1_a',epoch=completed,step=steps,global_samples_approx=world*samples,
                    unix_time_s=round(time.time(),6),elapsed_s=round(elapsed,5),val_loss=round(loss_total/n,7),
                    val_accuracy=round(accuracy,7),val_samples=n,val_s=round(time.perf_counter()-ev0,5),lr=round(lr,9),
                    best_val_accuracy=round(best,7),cumulative_samples_per_s=round(world*samples/max(elapsed,1e-9),3))
                vw.writerow(last_val);print('VALIDATION',json.dumps(last_val),flush=True)
                checkpoint=dict(model=base.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    epoch_completed=completed,step=steps,elapsed_s=elapsed,config=vars(a),world_size=world,best_val_accuracy=best)
                tmp=output/'checkpoint_latest.part';torch.save(checkpoint,tmp);os.replace(tmp,output/'checkpoint_latest.pt')
            validation_barrier_start=time.perf_counter();dist.barrier()
            sync_event(completed,steps,'post_validation_barrier',validation_barrier_start,'includes rank0 evaluation and checkpoint wait')
            stop=torch.tensor([int(a.max_seconds>0 and time.perf_counter()-begin>=a.max_seconds)],device=device)
            broadcast_start=time.perf_counter();dist.broadcast(stop,src=0)
            sync_event(completed,steps,'stop_decision_broadcast',broadcast_start)
            if stop.item():reason='time_budget';break
        if rank==0:
            summary=dict(experiment='E1_a',status='ok',reason=reason,world_size=world,completed_epochs=completed,
                steps_per_worker=steps,global_samples_approx=world*samples,wall_training_s=round(time.perf_counter()-begin,4),
                best_validation_accuracy=round(best,7),target_accuracy=a.target_accuracy,target_first_s=first_target,
                latest_validation=last_val,global_batch_size=world*a.batch_size,instrumented=True)
            (output/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
        final_barrier_start=time.perf_counter();dist.barrier()
        sync_event(completed,steps,'final_barrier',final_barrier_start)
    finally:
        # Stop only this host's monitor. A graceful stop writes monitor_finished_*.
        if tcp_monitor is not None:
            try:
                tcp_stop.touch()
                try:tcp_monitor.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    tcp_monitor.terminate()
                    try:tcp_monitor.wait(timeout=3)
                    except subprocess.TimeoutExpired:tcp_monitor.kill();tcp_monitor.wait()
            except Exception as e:
                print(f'WARNING: monitor shutdown failed: {e}',flush=True)
            finally:
                if tcp_monitor_log:tcp_monitor_log.close()
        sync_f.close()
        if monitoring:monitoring.close()
        steps_f.close();coll_f.close()
        if vf:vf.close()
        dist.destroy_process_group()

if __name__=='__main__':run()
