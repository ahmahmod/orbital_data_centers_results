# E2_FULL_METRICS_V1
#!/usr/bin/env python3
"""E3_1 explicit synchronous TCP ring: DDP-equivalent H=1 and fixed-H DiLoCo."""
import argparse,csv,datetime,json,os,pathlib,random,time,traceback,threading,queue
import torch
import torch.distributed as dist
import socket
import sys
from pathlib import Path
from e2_metrics import STEP_FIELDS, VALIDATION_FIELDS, writer, sync_cuda, evaluate_local, ResourceSampler
from torchvision import models
from e1_training import add_training_arguments, validate_training_args, loaders, lr_at
from e3_schedule import ContactScheduler
from communication.transport import RingTransport
from communication.ring_allreduce import RingAllReduce
from e3_startup import publish_ready, await_start, publish_failure

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


def main(methods=('ddp','diloco8','diloco64')):
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=methods,required=True)
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
    p.add_argument('--comm-timeout',type=int,default=900,help='Reactive TCP no-progress timeout in seconds')
    p.add_argument('--dataset',choices=['cifar100','synthetic'],default='cifar100')
    a=p.parse_args()
    validate_training_args(a)
    if a.world != 4:raise ValueError('E3 ring requires four ranks')
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
    schedule_mode=os.environ.get('E3_SCHEDULE_MODE','off')
    if schedule_mode not in ('off','aware'):
        raise ValueError('schedule mode off|aware')
    scheduler=ContactScheduler(os.environ['E3_TRACE'],a.start,a.duration,rank,a.method,output,
        mode=schedule_mode,margin=float(os.getenv('E3_RING_MARGIN_S','.5')),
        factor=float(os.getenv('E3_RING_FACTOR','2')),
        min_transfer=float(os.getenv('E3_RING_MIN_TRANSFER_S','.02')),
        run_id=os.environ.get('E3_RUN_ID',''),rate_scale=float(os.getenv('E3_RATE_SCALE','1')))
    if rank==0:print(f'[E3] mode={schedule_mode}; transport=decentralized_tcp_ring',flush=True)
    event_lock=threading.Lock()
    def log(event,step,bytes=0,peer='',detail=''):
        with event_lock:
            ev.writerow(dict(rank=rank,event=event,step=step,unix_s=time.time(),method=a.method,logical_bytes=bytes,peer=peer,detail=detail))
    progress,prog=writer(output/f'steps_rank{rank}.csv',STEP_FIELDS)
    valfile,valwriter=writer(output/f'validation_rank{rank}.csv',VALIDATION_FIELDS)
    iface=os.getenv('NCCL_SOCKET_IFNAME','') or os.getenv('E2_NET_IFACE','')
    resources=ResourceSampler(output,rank,iface,a.start,a.resource_interval_s) if iface else None
    next_validation=a.start + a.eval_interval_s if a.eval_interval_s>0 else float('inf')
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
    # Gloo is used ONLY before replay to establish identical initial weights.
    # No torch.distributed process group remains active during training.
    dist.init_process_group(backend='gloo',init_method='env://',world_size=a.world,rank=rank,
                            timeout=datetime.timedelta(seconds=a.comm_timeout))
    try:
        with torch.no_grad():
            for tensor in model.state_dict().values():
                host_tensor=tensor.detach().cpu().contiguous()
                dist.broadcast(host_tensor,src=0)
                tensor.copy_(host_tensor.to(device))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    ring=RingTransport(rank,json.loads(os.environ['E3_RING_PEERS']),
        int(os.getenv('E3_RING_PORT',str(a.port+1))),os.environ.get('E3_RUN_ID',''),scheduler,
        frame_bytes=int(os.getenv('E3_RING_FRAME_BYTES','1048576')),
        margin=float(os.getenv('E3_RING_MARGIN_S','.5')),
        factor=float(os.getenv('E3_RING_FACTOR','2')),
        open_guard=float(os.getenv('E3_RING_OPEN_GUARD_S','.25')),
        min_transfer=float(os.getenv('E3_RING_MIN_TRANSFER_S','.02')),
        connect_timeout=float(os.getenv('E3_RING_CONNECT_TIMEOUT_S','180')),
        io_timeout=a.comm_timeout,
        drain_guard=float(os.getenv('E3_RING_DRAIN_GUARD_S','2')),
        drain_timeout=float(os.getenv('E3_RING_DRAIN_TIMEOUT_S','240')))
    ring.connect()
    reducer=RingAllReduce(ring)
    print(f'[Rank {rank}] Ring TX->{ring.next} RX<-{ring.prev} connected; '
          f'mode={schedule_mode}; frame_bytes={ring.frame_bytes}',flush=True)

    def reduce_tensor(vector,step,tag):
        """Training thread: immutable snapshot, enqueue exchanges, await mean."""
        ready=time.time()
        opid=f'{rank}-{tag}-{step}'
        logical_bytes=vector.numel()*vector.element_size()
        log('comm_ready',step,logical_bytes,detail='tcp_ring_'+tag)
        host=vector.detach().to(device='cpu',dtype=torch.float32).contiguous().numpy()
        scheduler.record(comm_id=opid,current_step=step,scope='collective',
                         logical_bytes=logical_bytes,comm_ready_unix=ready,
                         comm_start_unix=time.time(),scheduler_action='start',
                         communication_status='started',rendezvous_mode='local_link_schedule',
                         note='gate_tag='+tag,current_H=sync_every,selected_H=sync_every)
        try:
            result,stop=reducer.reduce_mean(host,step)
        except Exception:
            scheduler.complete(opid,step,logical_bytes,status='failed',note='tcp_ring_failure')
            raise
        if stop:
            scheduler.record(comm_id=opid,current_step=step,scope='collective',
                scheduler_action='terminal_vote',communication_status='terminal',
                defer_duration_s=ring.last_collective_wait_s,note='gate_tag=terminal_vote')
            log('ring_terminal_vote',step,detail='global_wall_budget_stop_no_update')
            return None,opid
        out_tensor=torch.from_numpy(result).to(device=vector.device,dtype=vector.dtype)
        sync_cuda(torch)
        scheduler.record(comm_id=opid,current_step=step,scope='collective',
            scheduler_action='complete',communication_status='completed',
            comm_complete_unix=time.time(),logical_bytes=logical_bytes,
            defer_duration_s=ring.last_collective_wait_s,note='gate_tag='+tag)
        log('comm_complete',step,logical_bytes,
            detail=f'tcp_ring_{tag};tx_payload={ring.last_collective_tx_bytes};rx_payload={ring.last_collective_rx_bytes}')
        return out_tensor,opid
    # Initial model broadcast and both socket connections finish before replay.
    startup_run_id=os.environ.get('E3_RUN_ID','')
    if startup_run_id:
        publish_ready(output,rank,startup_run_id)
        log('worker_ready',0,detail='startup_barrier_ready')
        new_start=await_start(output,startup_run_id,
             timeout_s=float(os.environ.get('E3_STARTUP_READY_TIMEOUT','1800')))
        old_start=a.start
        a.start=new_start
        scheduler.start=new_start
        scheduler.end=new_start+a.duration
        ring.set_epoch(new_start,a.duration)
        next_validation=new_start+a.eval_interval_s if a.eval_interval_s>0 else float('inf')
        log('shared_start_received',0,detail=f'old={old_start:.6f};new={new_start:.6f}')
    else:
        log('worker_ready',0,detail='legacy_fixed_start_no_run_id')
    while time.time()<a.start:time.sleep(min(.05,a.start-time.time()))
    if resources:
        resources.start=a.start
        resources.start_sampling()
    start=time.time();step=0;local_samples=0
    comm_events=0;comm_bytes=0;comm_seconds=0.0;comm_max=0.0;comm_failures=0
    sync_every=int(''.join(c for c in a.method if c.isdigit()) or '1')
    baseline=flatten_model(model).clone() if a.method!='ddp' else None
    outer_m=torch.zeros_like(baseline) if baseline is not None else None
    outer_round=0
    try:
        # Fixed epochs or in-band common stop vote at the synchronization boundary.
        reason='epochs_completed'
        while epoch < a.epochs:
            wait_start=time.perf_counter()
            try:x,y=next(data_iter)
            except StopIteration:
                epoch+=1
                log('epoch_finished',step,detail=f'epoch={epoch}')
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
            backward_t0=time.perf_counter()
            scaler.scale(loss).backward()
            sync_cuda(torch);t_backward=time.perf_counter()
            if a.method=='ddp':
                # Synchronous data parallel: same global scaled gradient on
                # every worker before optimizer/GradScaler update.
                params=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None for p in params):
                    raise RuntimeError('Missing gradient in ring DDP')
                comm_t0=time.perf_counter()
                gradients=torch.cat([p.grad.detach().reshape(-1) for p in params])
                logical=gradients.numel()*gradients.element_size()
                reduced,ddp_opid=reduce_tensor(gradients,step,'gradient')
                if reduced is None:
                    reason='wall_budget'
                    break
                offset=0
                for par in params:
                    n=par.numel()
                    par.grad.copy_(reduced[offset:offset+n].view_as(par))
                    offset+=n
                sync_cuda(torch)
                comm_dt=time.perf_counter()-comm_t0
                comm_events+=1;comm_bytes+=logical;comm_seconds+=comm_dt
                comm_max=max(comm_max,comm_dt)
            t_before_opt=time.perf_counter()
            scaler.step(opt);scaler.update();sync_cuda(torch);t_optimizer=time.perf_counter()
            step+=1;local_samples+=len(y)
            log('local_optimizer_step',step)
            if a.method=='ddp':scheduler.record(comm_id=ddp_opid,current_step=step,model_apply_unix=time.time(),scheduler_action='model_apply',communication_status='applied',note='DDP optimizer step completed')
            stop_after_step=False
            if a.method in ('diloco8','diloco64') and step%sync_every==0:
                current=flatten_model(model)
                local_delta=current-baseline
                comm_t0=time.perf_counter()
                reduced,outer_opid=reduce_tensor(local_delta,step,'outer')
                if reduced is None:
                    reason='wall_budget'
                    log('scheduler_horizon_stop',step,detail='pending_outer_round_not_applied')
                    stop_after_step=True
                else:
                    local_delta=reduced
                    comm_dt=time.perf_counter()-comm_t0
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
                    comm_events+=1;comm_bytes+=local_delta.numel()*local_delta.element_size();comm_seconds+=comm_dt;comm_max=max(comm_max,comm_dt)
                    log('model_apply',step,detail='outer_sgd')
                    scheduler.record(comm_id=outer_opid,current_step=step,model_apply_unix=time.time(),scheduler_action='model_apply',communication_status='applied')
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
            if stop_after_step:
                break
            now=time.time()
            if now<scheduler.end and a.eval_interval_s>0 and now>=next_validation:
                acc,v_loss,n,v_sec=evaluate_local(model,testloader,device,torch,amp=amp)
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
        if epoch < a.epochs and reason=='epochs_completed':
            reason='wall_budget'
        log('run_finished',step,detail=reason)
        print(f'[Rank {rank}] Training finished at step={step}; epochs_completed={min(epoch, a.epochs)}; reason={reason}',flush=True)
    except Exception:
        log('run_failed',step,detail=traceback.format_exc()[-1500:])
        try:
            if os.environ.get('E3_RUN_ID'):
                detail=traceback.format_exc()[-1500:]
                publish_failure(output,rank,os.environ['E3_RUN_ID'],detail)
        except Exception:
            pass
        raise
    finally:
        if resources:resources.close()
        eventfile.close();progress.close();valfile.close()
        ring.close()
        scheduler.close()
        print(f'[Rank {rank}] Ring TX/RX threads closed',flush=True)
if __name__=='__main__':main()
