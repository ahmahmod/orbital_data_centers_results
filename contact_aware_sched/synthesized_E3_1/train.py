# E2_FULL_METRICS_V1
#!/usr/bin/env python3
"""Self-contained E2 PyTorch distributed ResNet18/CIFAR100 workload.

Algorithms: DDP (all-reduce each step), ring Gossip every K steps,
DiLoCo-style local-SGD with outer momentum every K, and FedAvg every K.
This is a standalone FedAvg implementation, NOT the Flower framework code.
Contact plan is NEVER an input to training; it is replayed independently.
"""
import argparse,csv,datetime,json,os,pathlib,random,time,traceback,threading,queue
import torch
import torch.distributed as dist
import socket
from e2_metrics import STEP_FIELDS, VALIDATION_FIELDS, writer, sync_cuda, evaluate_local, ResourceSampler
from torchvision import models
from e1_training import add_training_arguments, validate_training_args, loaders, lr_at
from e3_schedule import ContactScheduler

def emit(writer,rank,event,step,**extra):
    writer.writerow(dict(rank=rank,event=event,step=step,unix_s=time.time(),**extra))

def flatten_model(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

@torch.no_grad()
def set_model(model,vector):
    offset=0
    for p in model.parameters():
        n=p.numel();p.copy_(vector[offset:offset+n].reshape(p.shape));offset+=n

def evaluate(model,loader,device):
    model.eval();n=0;correct=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            pred=model(x).argmax(1);n+=len(y);correct+=(pred==y).sum().item()
    model.train()
    return correct/max(n,1)


class AsyncRing:
    """E1_b's bounded asynchronous Gloo ring with distinct send/recv threads."""
    def __init__(self, group, num_params, rank, world, scheduler=None):
        self.group = group
        self.scheduler = scheduler
        self.n = num_params
        self.left, self.right = (rank-1) % world, (rank+1) % world
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
                    self.events.put(('comm_error', 0, 0, repr(exc)))
                return
            version, payload, created_unix = item
            try:
                if self.scheduler is not None:
                    ok, opid=self.scheduler.gate('peer',version,payload.numel()*payload.element_size(),peer=self.right,version=version,created_unix=created_unix)
                    if not ok:
                        self.events.put(('comm_deferred_out_of_horizon',version,0,'no_future_contact'))
                        continue
                dist.send(payload, dst=self.right, group=self.group)
                if self.scheduler is not None:
                    self.scheduler.complete(opid,version,bytes_=payload.numel()*payload.element_size(),peer=self.right,version=version)
                self.events.put(('comm_complete', version,
                                 payload.numel()*payload.element_size(), 'ring_gossip'))
            except Exception as exc:
                self.events.put(('comm_error', version, 0, repr(exc)))
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
                    self.inbox.get_nowait()
                    self.events.put(('inbox_drop', version, 0, 'stale_snapshot_dropped'))
                    self.inbox.put_nowait(item)
        except Exception as exc:
            self.events.put(('comm_error', 0, 0, repr(exc)))

    def submit(self, version, payload):
        try:
            self.outbox.put_nowait((version, payload, time.time()))
            return True
        except queue.Full:
            return False

    def shutdown(self, timeout=30):
        # Do not indefinitely block the coordinator if a link is down.
        try:
            self.outbox.put_nowait(None)
        except queue.Full:
            return False
        self.sender.join(timeout)
        self.receiver.join(timeout)
        return not (self.sender.is_alive() or self.receiver.is_alive())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['ddp','gossip8','gossip64','diloco8','diloco64','fl8','fl64'],required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=4)
    p.add_argument('--master',required=True);p.add_argument('--port',type=int,default=29527)
    p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=int,default=3600)
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    add_training_arguments(p)
    p.add_argument('--outer-lr',type=float,default=0.3);p.add_argument('--outer-momentum',type=float,default=.9)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--eval-interval-s',type=float,default=float(os.getenv('E2_EVAL_INTERVAL_S','300')))
    p.add_argument('--resource-interval-s',type=float,default=float(os.getenv('E2_RESOURCE_INTERVAL_S','2')))
    p.add_argument('--comm-timeout',type=int,default=900,help='NCCL operation timeout in seconds')
    p.add_argument('--dataset',choices=['cifar100','synthetic'],default='cifar100')
    a=p.parse_args()
    validate_training_args(a)
    if a.comm_timeout<=0:raise ValueError('comm-timeout must be positive')
    if not (0 < a.outer_lr <= 1):raise ValueError('outer-lr must be in (0, 1]')
    if not (0 <= a.outer_momentum < 1):raise ValueError('outer-momentum must be in [0, 1)')
    if a.eval_every<1:raise ValueError('--eval-every must be positive')
    os.environ['MASTER_ADDR']=a.master;os.environ['MASTER_PORT']=str(a.port)
    torch.manual_seed(a.seed);random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True
    rank=a.rank;device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('E2 requires real CUDA GPU on each node')
    torch.cuda.set_device(0)
    output=pathlib.Path(a.output);output.mkdir(parents=True,exist_ok=True)
    eventfile=(output/f'e2_events_rank{rank}.csv').open('w',newline='',buffering=1)
    ev=csv.DictWriter(eventfile,fieldnames=['rank','event','step','unix_s','method','logical_bytes','peer','detail'])
    ev.writeheader()
    scheduler=ContactScheduler(os.environ['E3_TRACE'], a.start, a.duration, rank, a.method, output,
        mode=os.environ.get('E3_SCHEDULE_MODE','off'),margin=float(os.environ.get('E3_MARGIN_S','15')),
        factor=float(os.environ.get('E3_ESTIMATE_FACTOR','2')),
        run_id=os.environ.get('E3_RUN_ID',''),rate_scale=float(os.environ.get('E3_RATE_SCALE','1')))
    event_lock=threading.Lock()
    def log(event,step,bytes=0,peer='',detail=''):
        with event_lock:
            ev.writerow(dict(rank=rank,event=event,step=step,unix_s=time.time(),method=a.method,logical_bytes=bytes,peer=peer,detail=detail))
    progress,prog=writer(output/f'steps_rank{rank}.csv',STEP_FIELDS)
    valfile,valwriter=writer(output/f'validation_rank{rank}.csv',VALIDATION_FIELDS)
    iface=os.getenv('NCCL_SOCKET_IFNAME','') or os.getenv('E2_NET_IFACE','')
    resources=ResourceSampler(output,rank,iface,a.start,a.resource_interval_s) if iface else None
    if resources:resources.start_sampling()
    next_validation=a.start + a.eval_interval_s if a.eval_interval_s>0 else float('inf')
    e1b_gossip=a.method.startswith('gossip')
    model=models.resnet18(num_classes=100)
    model.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool=torch.nn.Identity()
    model=model.to(device)
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay)
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    amp=not a.no_amp
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    loader,testloader,sampler,train_n,val_n=loaders(a,rank,a.world)
    epoch=0
    sampler.set_epoch(epoch)
    data_iter=iter(loader)
    dist.init_process_group(backend='nccl',init_method='env://',world_size=a.world,rank=rank,
         timeout=datetime.timedelta(seconds=a.comm_timeout))
    # E1_a-equivalent native DDP: bucketed gradient all-reduce at backward.
    if a.method=='ddp':
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],output_device=0)
    else:
        for par in model.parameters():dist.broadcast(par.data,src=0)
    dist.barrier();log('worker_ready',0)
    while time.time()<a.start:time.sleep(min(.05,a.start-time.time()))
    start=time.time();step=0;local_samples=0
    comm_events=0;comm_bytes=0;comm_seconds=0.0;comm_max=0.0;comm_failures=0
    sync_every=int(''.join(c for c in a.method if c.isdigit()) or '1')
    baseline=flatten_model(model).clone()
    outer_m=torch.zeros_like(baseline)
    outer_round=0
    # E1_b: CPU/Gloo exchange using bounded send and receive queues.
    gossip_pg=dist.new_group(ranks=list(range(a.world)),backend='gloo') if e1b_gossip else None
    ring=AsyncRing(gossip_pg, baseline.numel(), rank, a.world, scheduler=scheduler) if e1b_gossip else None

    def communicate_gossip(step, send=True):
        if ring is None:
            return
        # Drain completed sends and errors from the background sender.
        while True:
            try:
                kind,version,payload,detail=ring.events.get_nowait()
            except queue.Empty:
                break
            log(kind,version,payload,ring.right,detail)
        # Consume the latest received model snapshots without blocking training.
        while True:
            try:
                neighbor_version,neighbor_weights=ring.inbox.get_nowait()
            except queue.Empty:
                break
            own=flatten_model(model)
            set_model(model,.5*own+.5*neighbor_weights.to(device))
            log('model_apply',step,detail=(
                f'gossip_from_rank={ring.left};received_version={neighbor_version};'
                f'step_difference={step-neighbor_version}'))
        if send and step%sync_every==0:
            payload=torch.cat((torch.tensor([float(step)],dtype=torch.float32),
                               flatten_model(model).cpu()))
            if ring.submit(step,payload):
                log('comm_ready',step,payload.numel()*payload.element_size(),ring.right,'ring_gossip')
            else:
                log('comm_skipped_busy',step,0,ring.right,'outbox_full')

    def validate_at_epoch(step,completed_epoch):
        # E1_b: pre/post validation barriers, but never average model weights here.
        communicate_gossip(step,send=False)
        ok,pre_id=scheduler.gate('collective',step,4,tag=f'preval_{completed_epoch}')
        if not ok:raise RuntimeError('E3: no feasible contact for pre-validation barrier')
        dist.barrier()
        scheduler.complete(pre_id,step,4)
        log('pre_validation_barrier',step,detail=f'epoch={completed_epoch}')
        if completed_epoch % a.eval_every == 0 or completed_epoch == a.epochs:
            acc,v_loss,n,v_sec=evaluate_local(model,testloader,device,torch,amp=amp)
            stamp=time.time()
            valwriter.writerow(dict(experiment='E2',method=a.method,rank=rank,epoch=completed_epoch,
                step=step,unix_time_s=stamp,elapsed_s=stamp-a.start,val_loss=v_loss,
                val_accuracy=acc,val_samples=n,val_s=v_sec,local_samples=local_samples,
                global_samples_approx=local_samples*a.world))
            if rank==0:
                with (output/'validation.csv').open('a',newline='') as legacy:
                    lw=csv.writer(legacy)
                    if legacy.tell()==0:
                        lw.writerow(['unix_s','elapsed_s','step','accuracy','method','val_accuracy','val_loss'])
                    lw.writerow([stamp,stamp-a.start,step,acc,a.method,acc,v_loss])
        ok,post_id=scheduler.gate('collective',step,4,tag=f'postval_{completed_epoch}')
        if not ok:raise RuntimeError('E3: no feasible contact for post-validation barrier')
        dist.barrier()
        scheduler.complete(post_id,step,4)
        log('post_validation_barrier',step,detail=f'epoch={completed_epoch}')
        # Rank 0 decides at an epoch boundary, as in E1_b.
        stop=torch.tensor([int(time.time()>=a.start+a.duration) if rank==0 else 0],
                          dtype=torch.int32,device=device)
        dist.broadcast(stop,src=0)
        return bool(stop.item())
    try:
        # Match E1_b: finish on epoch budget OR wall-clock budget, whichever comes first.
        reason='epochs_completed'
        while epoch < a.epochs and (e1b_gossip or time.time()<a.start+a.duration):
            wait_start=time.perf_counter()
            try:x,y=next(data_iter)
            except StopIteration:
                epoch+=1
                log('epoch_finished',step,detail=f'epoch={epoch}')
                if e1b_gossip and validate_at_epoch(step,epoch):
                    reason='wall_budget'
                    break
                if epoch >= a.epochs:
                    break
                sampler.set_epoch(epoch)
                data_iter=iter(loader)
                x,y=next(data_iter)
            step_t0=time.perf_counter(); batch_wait=step_t0-wait_start
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            sync_cuda(torch);t_h2d=time.perf_counter()
            current_lr=lr_at(min(epoch,a.epochs-1),a)
            for group in opt.param_groups:group['lr']=current_lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                logits=model(x);loss=criterion(logits,y)
            sync_cuda(torch);t_forward=time.perf_counter()
            if a.method=='ddp':
                logical=sum(p.numel()*p.element_size() for p in model.parameters())
                log('comm_ready',step,logical,detail='ddp_bucketed_backward')
                ok,ddp_opid=scheduler.gate('collective',step,logical)
                if not ok:raise RuntimeError('E3: no feasible all-worker contact for DDP gradient synchronization')
            backward_t0=time.perf_counter()
            scaler.scale(loss).backward()
            sync_cuda(torch);t_backward=time.perf_counter()
            if a.method=='ddp':
                # Approximate logical gradient payload. Backward includes NCCL.
                comm_dt=t_backward-backward_t0
                comm_events+=1;comm_bytes+=logical;comm_seconds+=comm_dt
                comm_max=max(comm_max,comm_dt)
                log('comm_complete',step,logical,detail='ddp_bucketed_backward')
                scheduler.complete(ddp_opid,step,logical)
            t_before_opt=time.perf_counter()
            scaler.step(opt);scaler.update();sync_cuda(torch);t_optimizer=time.perf_counter()
            step+=1;local_samples+=len(y)
            log('local_optimizer_step',step)
            if a.method=='ddp':scheduler.record(comm_id=ddp_opid,current_step=step,model_apply_unix=time.time(),scheduler_action='model_apply',communication_status='applied',note='DDP optimizer step completed')
            if a.method in ('fl8','fl64') and step%sync_every==0:
                weights=flatten_model(model)
                log('comm_ready',step,weights.numel()*weights.element_size(),detail='fedavg_weights')
                comm_t0=time.perf_counter()
                dist.all_reduce(weights,op=dist.ReduceOp.SUM);weights.div_(a.world)
                set_model(model,weights)
                sync_cuda(torch)
                comm_dt=time.perf_counter()-comm_t0
                comm_events+=1;comm_bytes+=weights.numel()*weights.element_size();comm_seconds+=comm_dt;comm_max=max(comm_max,comm_dt)
                log('comm_complete',step,weights.numel()*weights.element_size(),detail='fedavg_weights')
                log('model_apply',step,detail='fedavg')
            if a.method in ('diloco8','diloco64') and step%sync_every==0:
                current=flatten_model(model)
                local_delta=current-baseline
                log('comm_ready',step,local_delta.numel()*local_delta.element_size(),detail='diloco_delta')
                ok,outer_opid=scheduler.gate('collective',step,local_delta.numel()*local_delta.element_size())
                if not ok:raise RuntimeError('E3: no feasible contact for DiLoCo outer synchronization')
                comm_t0=time.perf_counter()
                dist.all_reduce(local_delta,op=dist.ReduceOp.SUM);local_delta.div_(a.world)
                outer_m.mul_(a.outer_momentum).add_(local_delta)
                baseline=baseline+a.outer_lr*outer_m
                outer_round+=1
                # Sample norms to diagnose an unstable outer optimizer.
                if outer_round == 1 or outer_round % 100 == 0:
                    norms=[float(v.item()) for v in (local_delta.norm(),outer_m.norm(),baseline.norm())]
                    log('outer_diagnostics',step,detail=(
                        f'round={outer_round};outer_lr={a.outer_lr};'
                        f'outer_momentum={a.outer_momentum};'
                        f'delta_norm={norms[0]:.6g};velocity_norm={norms[1]:.6g};'
                        f'anchor_norm={norms[2]:.6g}'))
                    if not all(__import__('math').isfinite(v) for v in norms):
                        raise FloatingPointError('Non-finite DiLoCo outer update')
                set_model(model,baseline)
                sync_cuda(torch)
                comm_dt=time.perf_counter()-comm_t0
                comm_events+=1;comm_bytes+=local_delta.numel()*local_delta.element_size();comm_seconds+=comm_dt;comm_max=max(comm_max,comm_dt)
                log('comm_complete',step,local_delta.numel()*local_delta.element_size(),detail='diloco_delta')
                log('model_apply',step,detail='outer_sgd')
                scheduler.complete(outer_opid,step,local_delta.numel()*local_delta.element_size(),applied=True)
            if e1b_gossip:
                communicate_gossip(step)
            sync_cuda(torch);step_end=time.perf_counter()
            now=time.time()
            # Gossip runs in a background thread: its durations are recorded in
            # e2_events_rank*.csv; never attribute them to a fabricated single step.
            prog.writerow(dict(experiment='E2',host=socket.gethostname(),rank=rank,local_rank=0,
                epoch=epoch+1,step=step,unix_s=now,unix_time_s=now,elapsed_s=now-a.start,
                loss=float(loss.detach()),lr=current_lr,local_samples=local_samples,
                global_samples_approx=local_samples*a.world,batch_samples=len(y),
                step_s=step_end-step_t0,batch_wait_s=batch_wait,h2d_s=t_h2d-step_t0,
                forward_s=t_forward-t_h2d,backward_with_comm_s=t_before_opt-t_forward,
                optimizer_s=t_optimizer-t_before_opt,other_s=step_end-t_optimizer,
                collectives=comm_events,collective_bytes_logical=comm_bytes,
                collective_sum_host_s=comm_seconds,collective_span_host_s=comm_seconds,
                collective_max_host_s=comm_max,collective_failures=comm_failures,
                gpu_memory_allocated_mb=torch.cuda.memory_allocated(device)/(1024**2),
                samples_per_s=len(y)/max(step_end-step_t0,1e-9),step_wall_s=step_end-wait_start))
            comm_events=0;comm_bytes=0;comm_seconds=0.0;comm_max=0.0;comm_failures=0
            if not e1b_gossip and a.eval_interval_s>0 and now>=next_validation:
                acc,v_loss,n,v_sec=evaluate_local(model.module if a.method=='ddp' else model,testloader,device,torch,amp=amp)
                stamp=time.time(); rec=dict(experiment='E2',method=a.method,rank=rank,epoch=epoch+1,
                    step=step,unix_time_s=stamp,elapsed_s=stamp-a.start,val_loss=v_loss,
                    val_accuracy=acc,val_samples=n,val_s=v_sec,local_samples=local_samples,
                    global_samples_approx=local_samples*a.world)
                valwriter.writerow(rec)
                if rank==0:
                    with (output/'validation.csv').open('a',newline='') as legacy:
                        lw=csv.writer(legacy)
                        if legacy.tell()==0:lw.writerow(['unix_s','elapsed_s','step','accuracy','method','val_accuracy','val_loss'])
                        lw.writerow([stamp,stamp-a.start,step,acc,a.method,acc,v_loss])
                next_validation += (int((stamp-next_validation)//a.eval_interval_s)+1)*a.eval_interval_s
            if rank == 0 and a.log_every > 0 and (step == 1 or step % a.log_every == 0):
                elapsed = time.time() - a.start
                current_loss = loss.detach().item()
                steps_per_sec = step / max(elapsed, 1e-6)

                print(
                    f"[{a.method.upper()}] "
                    f"Step={step} "
                    f"Loss={current_loss:.4f} "
                    f"Samples={local_samples} "
                    f"Elapsed={elapsed:.1f}s "
                    f"Speed={steps_per_sec:.2f} steps/s",
                    flush=True
                )
            # E2_FULL_METRICS: wall-clock validation runs independently on EVERY rank.
        if not e1b_gossip and epoch < a.epochs:
            reason='wall_budget'
        log('run_finished',step,detail=reason)
        print(f'[Rank {rank}] Training finished at step={step}; epochs_completed={min(epoch, a.epochs)}; reason={reason}',flush=True)
    except Exception:
        log('run_failed',step,detail=traceback.format_exc()[-1500:]);raise
    finally:
        print(f'[Rank {rank}] Cleanup started; gossip sender active={ring.sender.is_alive() if ring else False}',flush=True)
        if ring is not None:
            closed=ring.shutdown(timeout=30)
            if not closed:
                print(f'[Rank {rank}] WARNING: Gossip send/receive threads remain active',flush=True)
        if resources:resources.close()
        eventfile.close();progress.close();valfile.close()
        scheduler.close()
        print(f'[Rank {rank}] Destroying process group',flush=True)
        # Avoid a final blocking barrier on disconnected network.
        try:
            dist.destroy_process_group()
            print(f'[Rank {rank}] Process group destroyed',flush=True)
        except Exception as exc:
            print(f'[Rank {rank}] Process group shutdown error: {exc!r}',flush=True)
if __name__=='__main__':main()
