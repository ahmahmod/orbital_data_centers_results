# E2_FULL_METRICS_V1
#!/usr/bin/env python3
"""E3_1 synchronous NCCL workload: fixed-H DDP and DiLoCo."""
import argparse,csv,datetime,json,os,pathlib,random,time,traceback,threading,queue
import torch
import torch.distributed as dist
import socket
import sys
from pathlib import Path
from e2_metrics import STEP_FIELDS, VALIDATION_FIELDS, writer, sync_cuda, evaluate_local, ResourceSampler, close_loaders
from torchvision import models
from e1_training import add_training_arguments, validate_training_args, loaders, lr_at
from e3_batch_control import BatchScheduler
from e3_reactive import ReactiveScheduler
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
    p.add_argument('--comm-timeout',type=int,default=900,help='NCCL operation timeout in seconds')
    p.add_argument('--dataset',choices=['cifar100','synthetic'],default='cifar100')
    p.add_argument('--eval-steps',type=int,default=int(os.getenv('E3_EVAL_STEPS','128')))
    a=p.parse_args()
    validate_training_args(a)
    h=int(''.join(c for c in a.method if c.isdigit()) or '1')
    if a.eval_steps<1 or a.eval_steps%h:raise ValueError('eval-steps must be a positive multiple of H')
    if a.world != 4:raise ValueError('E3 NCCL control requires four ranks')
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
    scheduler_cls=ReactiveScheduler if schedule_mode=='off' else BatchScheduler
    scheduler_options={'comm_timeout':a.comm_timeout}
    scheduler=scheduler_cls(os.environ['E3_TRACE'], a.start, a.duration, rank, a.method, output,
        mode=schedule_mode,margin=float(os.environ.get('E3_MARGIN_S','15')),
        factor=float(os.environ.get('E3_ESTIMATE_FACTOR','2')),
        run_id=os.environ.get('E3_RUN_ID',''),rate_scale=float(os.environ.get('E3_RATE_SCALE','1')),
        **scheduler_options)
    if rank==0:
        print(f'[E3] mode={schedule_mode}; scheduler={scheduler_cls.__name__}',flush=True)
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
    dist.init_process_group(backend='nccl',init_method='env://',world_size=a.world,rank=rank,
         device_id=torch.device('cuda:0'),timeout=datetime.timedelta(seconds=a.comm_timeout))
    # Same native DDP in reactive, per-operation and batched aware modes.
    # Keep BatchNorm buffers local, as in the preceding E3 DDP policy.
    if a.method=='ddp':
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],output_device=0,broadcast_buffers=False)
    else:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor,src=0)
    dist.barrier(device_ids=[0])
    # Initialize DDP/CIFAR-100 before beginning the shared contact timeline.
    validation_model=model.module if a.method=='ddp' else model
    initial_metrics=evaluate_local(validation_model,testloader,device,torch,amp=amp)
    initial_stamp=time.time()
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
        if schedule_mode=='aware' and scheduler.broker:
            topology=scheduler.broker.read('startup/topology')
            if not topology or topology.get('run_id')!=startup_run_id:
                raise RuntimeError('Missing coordinator topology decision')
            scheduler.set_topology(topology)
            if rank==0:print(f'[E3] topology={scheduler.topology}; required_edges={len(scheduler.required_edges)}',flush=True)
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
    total_steps=len(loader)*a.epochs
    logical=sum(p.numel()*p.element_size() for p in model.parameters())
    baseline=flatten_model(model).clone() if a.method!='ddp' else None
    outer_m=torch.zeros_like(baseline) if baseline is not None else None
    outer_round=0
    last_validated=-1
    def validate_checkpoint(kind,metrics=None,evaluated=None):
        nonlocal last_validated
        acc,v_loss,n,v_sec=metrics if metrics is not None else evaluate_local(
            validation_model,testloader,device,torch,amp=amp)
        stamp=time.time() if evaluated is None else evaluated
        # Initial model was measured before readiness; place it at trace t=0.
        recorded=a.start if kind=='initial' else stamp
        rec=dict(experiment='E3_1',method=a.method,rank=rank,
            epoch=step/len(loader),completed_epochs=step//len(loader),
            step=step,unix_time_s=recorded,elapsed_s=recorded-a.start,
            val_loss=v_loss,val_accuracy=acc,val_samples=n,val_s=v_sec,
            local_samples=local_samples,global_samples_approx=local_samples*a.world,
            validation_kind=kind,evaluated_unix_s=stamp,outer_round=outer_round,
            within_horizon=int(recorded<=scheduler.end))
        valwriter.writerow(rec)
        if rank==0:
            with (output/'validation.csv').open('a',newline='') as legacy:
                lw=csv.writer(legacy)
                if legacy.tell()==0:lw.writerow(['unix_s','elapsed_s','step','accuracy','method','val_accuracy','val_loss'])
                lw.writerow([recorded,recorded-a.start,step,acc,a.method,acc,v_loss])
        last_validated=step
    try:
        validate_checkpoint('initial',initial_metrics,initial_stamp)
        # Match E1_b: finish on epoch budget OR wall-clock budget, whichever comes first.
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
            finished_op=None
            if a.method=='ddp':
                # Gate before forward: the reducer can communicate there.
                log('comm_ready',step,logical,detail='ddp_native_step')
                ok,ddp_opid=scheduler.gate('collective',step,logical,
                    remaining_ops=min(total_steps-step,a.eval_steps-step%a.eval_steps),tag='gradient')
                if not ok:
                    reason='wall_budget' if schedule_mode=='off' else 'no_feasible_contact_or_wall_budget'
                    log('scheduler_horizon_stop',step,detail=reason)
                    break
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                logits=model(x);loss=criterion(logits,y)
            sync_cuda(torch);t_forward=time.perf_counter()
            backward_t0=time.perf_counter()
            scaler.scale(loss).backward()
            sync_cuda(torch);t_backward=time.perf_counter()
            if a.method=='ddp':
                # Native bucketing and overlap remain intact. Backward time
                # contains compute + NCCL; it is not pure wire transfer time.
                comm_dt=t_backward-backward_t0
                comm_events+=1;comm_bytes+=logical;comm_seconds+=comm_dt
                comm_max=max(comm_max,comm_dt)
                log('comm_complete',step,logical,detail='ddp_bucketed_backward')
                scheduler.complete(ddp_opid,step,logical)
                finished_op=(ddp_opid,step)
            t_before_opt=time.perf_counter()
            scaler.step(opt);scaler.update();sync_cuda(torch);t_optimizer=time.perf_counter()
            step+=1;local_samples+=len(y)
            log('local_optimizer_step',step)
            if a.method=='ddp':scheduler.record(comm_id=ddp_opid,current_step=step,model_apply_unix=time.time(),scheduler_action='model_apply',communication_status='applied',note='DDP optimizer step completed')
            stop_after_step=False
            if a.method in ('diloco8','diloco64') and step%sync_every==0:
                current=flatten_model(model)
                local_delta=current-baseline
                log('comm_ready',step,local_delta.numel()*local_delta.element_size(),detail='diloco_delta')
                sync_cuda(torch)
                ok,outer_opid=scheduler.gate('collective',step,local_delta.numel()*local_delta.element_size(),
                    remaining_ops=min(1+(total_steps-step)//sync_every,
                        1+((a.eval_steps-step%a.eval_steps)%a.eval_steps)//sync_every),tag='outer')
                if not ok:
                    reason='wall_budget' if schedule_mode=='off' else 'no_feasible_contact_or_wall_budget'
                    log('scheduler_horizon_stop',step,detail=reason+'; pending_outer_round_not_applied')
                    stop_after_step=True
                else:
                    comm_t0=time.perf_counter()
                    dist.all_reduce(local_delta,op=dist.ReduceOp.SUM);local_delta.div_(a.world)
                    sync_cuda(torch)
                    comm_dt=time.perf_counter()-comm_t0
                    scheduler.complete(outer_opid,step,local_delta.numel()*local_delta.element_size())
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
                    log('comm_complete',step,local_delta.numel()*local_delta.element_size(),detail='diloco_delta')
                    log('model_apply',step,detail='outer_sgd')
                    scheduler.record(comm_id=outer_opid,current_step=step,model_apply_unix=time.time(),scheduler_action='model_apply',communication_status='applied')
                    finished_op=(outer_opid,step)
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
            # Validation is deliberately deferred to a drained batch boundary.
            # Otherwise an independent validation could exhaust a granted window.
            if finished_op is not None:
                scheduler.finish_operation(*finished_op)
            now=time.time()
            if step%a.eval_steps==0 and not scheduler.in_batch and now<scheduler.end:
                validate_checkpoint('periodic')
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
        if scheduler.in_batch:
            raise RuntimeError('Training ended with undrained collective grant')
        if epoch < a.epochs and reason=='epochs_completed':
            reason='wall_budget'
        if last_validated!=step:
            validate_checkpoint('final')
        log('run_finished',step,detail=reason)
        print(f'[Rank {rank}] Training finished at step={step}; epochs_completed={min(epoch, a.epochs)}; reason={reason}',flush=True)
    except Exception:
        log('run_failed',step,detail=traceback.format_exc()[-1500:])
        try:
            if os.environ.get('E3_RUN_ID'):
                detail=traceback.format_exc()[-1500:]
                if scheduler.broker:
                    try:scheduler.broker.write(f'failure/rank{rank}',dict(rank=rank,detail=detail,run_id=os.environ['E3_RUN_ID']))
                    except Exception:pass
                publish_failure(output,rank,os.environ['E3_RUN_ID'],detail)
        except Exception:
            pass
        raise
    finally:
        if resources:resources.close()
        eventfile.close();progress.close();valfile.close()
        close_loaders(loader,testloader)
        scheduler.close()
        print(f'[Rank {rank}] Destroying process group',flush=True)
        # Avoid a final blocking barrier on disconnected network.
        try:
            dist.destroy_process_group()
            print(f'[Rank {rank}] Process group destroyed',flush=True)
        except Exception as exc:
            print(f'[Rank {rank}] Process group shutdown error: {exc!r}',flush=True)
            raise
    from e3_startup import atomic_json
    atomic_json(output/f'e3_trainer_done_rank{rank}.json',dict(
        run_id=os.environ.get('E3_RUN_ID',''),rank=rank,step=step,
        completed_epochs=step//len(loader),outer_round=outer_round,
        reason=reason,finished_unix=time.time()))
if __name__=='__main__':main()
