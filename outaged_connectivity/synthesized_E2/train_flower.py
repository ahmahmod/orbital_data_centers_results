# E2_FULL_METRICS_V1
#!/usr/bin/env python3
"""Flower 1.24 FedAvg reference implementation (4 clients, 1 server on worker 0).
Legacy start_server/start_client API is used deliberately for reproducibility.
Rank-0 local client shares a host with the server and bypasses tc for loopback.
"""
import argparse,csv,json,os,pathlib,random,time
import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import torch
from torchvision import models
from e1_training import add_training_arguments, validate_training_args, loaders, lr_at
from e2_metrics import STEP_FIELDS, VALIDATION_FIELDS, writer, sync_cuda, evaluate_local, ResourceSampler
import socket

def build_model():
    model=models.resnet18(num_classes=100)
    model.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool=torch.nn.Identity()
    return model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--role',choices=['server','client'],required=True)
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--server',required=True)
    ap.add_argument('--start',type=float,required=True)
    ap.add_argument('--duration',type=int,default=3600)
    ap.add_argument('--steps-per-round',type=int,choices=[8,64],required=True)
    ap.add_argument('--rounds',type=int,default=500,help='E1_b launcher default: 500 FedAvg rounds')
    ap.add_argument('--data-root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--dataset',choices=['cifar100','synthetic'],default='cifar100')
    add_training_arguments(ap)
    ap.add_argument('--eval-interval-s',type=float,default=float(os.getenv('E2_EVAL_INTERVAL_S','300')))
    ap.add_argument('--resource-interval-s',type=float,default=float(os.getenv('E2_RESOURCE_INTERVAL_S','2')))
    ap.add_argument('--round-timeout',type=float,default=900,help='Flower round timeout in seconds')
    a=ap.parse_args()
    validate_training_args(a)
    if a.round_timeout<=0:raise ValueError('round-timeout must be positive')
    if a.rounds<=0:raise ValueError('rounds must be positive')
    while time.time()<a.start:time.sleep(min(.1,max(.005,a.start-time.time())))
    if a.role=='server':
        # Match flower_E1_b: fixed rounds, all four participants for both fit
        # and global-model evaluation. Reject incomplete rounds.
        out=pathlib.Path(a.output)
        out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)
        initial=[v.detach().cpu().numpy().copy()
                 for v in build_model().state_dict().values()]
        class FedAvgMeter(fl.server.strategy.FedAvg):
            def __init__(self):
                super().__init__(
                    fraction_fit=1.0,fraction_evaluate=1.0,
                    min_fit_clients=4,min_evaluate_clients=4,min_available_clients=4,
                    accept_failures=False,initial_parameters=ndarrays_to_parameters(initial),
                    on_fit_config_fn=lambda rnd:{'round':rnd},
                    on_evaluate_config_fn=lambda rnd:{'round':rnd})
                self.completed_rounds=0
                self.eval_rounds=0
                self.best=0.0
                self.started=time.time()
                self.validation_fp=(out/'validation_aggregate.csv').open('w',newline='',buffering=1)
                self.validation=csv.writer(self.validation_fp)
                self.validation.writerow(['experiment','method','epoch','step','elapsed_s','unix_time_s',
                    'mean_accuracy','min_accuracy','max_accuracy','std_accuracy',
                    'mean_loss','min_loss','max_loss','std_loss','val_samples_per_worker',
                    'world_size','mean_accuracy_target_reached','all_workers_target_reached',
                    'observed_ranks','complete'])
                self.legacy_fp=(out/'validation.csv').open('w',newline='',buffering=1)
                self.legacy=csv.writer(self.legacy_fp)
                self.legacy.writerow(['unix_s','elapsed_s','step','accuracy','method','val_accuracy','val_loss'])
            def aggregate_fit(self,server_round,results,failures):
                if failures or len(results)!=4:
                    raise RuntimeError(f'FedAvg round {server_round}: expected 4 fit clients; got {len(results)}')
                params,metrics=super().aggregate_fit(server_round,results,failures)
                if params is None:
                    raise RuntimeError(f'FedAvg round {server_round}: empty aggregate')
                self.completed_rounds=server_round
                return params,metrics
            def aggregate_evaluate(self,server_round,results,failures):
                if failures or len(results)!=4:
                    raise RuntimeError(f'FedAvg round {server_round}: expected 4 evaluations; got {len(results)}')
                loss,metrics=super().aggregate_evaluate(server_round,results,failures)
                if loss is None:
                    raise RuntimeError(f'FedAvg round {server_round}: no aggregate evaluation')
                ordered=sorted(results,key=lambda item:int(item[1].metrics['worker_rank']))
                acc=np.array([float(res.metrics['accuracy']) for _,res in ordered])
                losses=np.array([float(res.loss) for _,res in ordered])
                sizes=[int(res.num_examples) for _,res in ordered]
                if len(set(sizes))!=1:
                    raise RuntimeError('Flower validation holdout sizes differ among workers')
                mean=float(acc.mean())
                mean_loss=float(losses.mean())
                self.best=max(self.best,mean)
                now=time.time()
                elapsed=now-a.start
                self.validation.writerow(['E2',f'fl{a.steps_per_round}',server_round,server_round,
                    elapsed,now,mean,float(acc.min()),float(acc.max()),float(acc.std()),
                    mean_loss,float(losses.min()),float(losses.max()),float(losses.std()),
                    sizes[0],4,int(mean>=.7),int(float(acc.min())>=.7),'0,1,2,3',1])
                self.legacy.writerow([now,elapsed,server_round,mean,
                    f'fl{a.steps_per_round}',mean,mean_loss])
                self.eval_rounds=server_round
                print(f'FLOWER ROUND {server_round}/{a.rounds}: global_accuracy={mean:.4f}',flush=True)
                return loss,{'accuracy':mean}
            def close(self):
                self.validation_fp.close()
                self.legacy_fp.close()
                (out/'summary.json').write_text(json.dumps(dict(
                    experiment='E2',method=f'fl{a.steps_per_round}',
                    status='ok' if self.completed_rounds==a.rounds and self.eval_rounds==a.rounds else 'incomplete',
                    completed_rounds=self.completed_rounds,evaluated_rounds=self.eval_rounds,
                    requested_rounds=a.rounds,best_validation_accuracy=self.best,
                    world_size=4,local_steps=a.steps_per_round,
                    training_stop='fixed Flower round count; duration is coordinator safety deadline'),indent=2))
        strategy=FedAvgMeter()
        try:
            fl.server.start_server(server_address=a.server,
                config=fl.server.ServerConfig(num_rounds=a.rounds,round_timeout=a.round_timeout),
                strategy=strategy,grpc_max_message_length=2147483647)
        finally:
            strategy.close()
        if strategy.completed_rounds!=a.rounds or strategy.eval_rounds!=a.rounds:
            raise RuntimeError(f'Flower incomplete: fit={strategy.completed_rounds}, evaluated={strategy.eval_rounds}, requested={a.rounds}')
        return
    rank=a.rank
    torch.manual_seed(a.seed);random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True
    if not torch.cuda.is_available():raise RuntimeError('Flower client requires GPU')
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    model=build_model().to(device)
    amp=not a.no_amp
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    loader,val_loader,sampler,train_n,val_n=loaders(a,rank,4)
    epoch=0
    sampler.set_epoch(epoch)
    iterator=iter(loader)
    opt=None
    scaler=None
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    fe=(out/f'e2_events_rank{rank}.csv').open('w',newline='',buffering=1)
    wr=csv.DictWriter(fe,fieldnames=['rank','event','step','unix_s','method','logical_bytes','peer','detail']);wr.writeheader()
    fp,prog=writer(out/f'steps_rank{rank}.csv',STEP_FIELDS)
    vf,vw=writer(out/f'validation_rank{rank}.csv',VALIDATION_FIELDS)
    iface=os.getenv('E2_NET_IFACE','')
    resources=ResourceSampler(out,rank,iface,a.start,a.resource_interval_s) if iface else None
    if resources:resources.start_sampling()
    # E1_b evaluates the aggregated global model through Flower every round.
    step=0;round_num=0;total_samples=0;previous_ready_step=None
    def emit(event,bytes=0,detail=''):
        wr.writerow({'rank':rank,'event':event,'step':step,'unix_s':time.time(),
                    'method':f'fl{a.steps_per_round}','logical_bytes':bytes,'peer':'server',
                    'detail':detail})
    def get_params():return [v.detach().cpu().numpy() for v in model.state_dict().values()]
    def set_params(params):
        keys=list(model.state_dict().keys())
        converted={key:torch.as_tensor(value) for key,value in zip(keys,params)}
        model.load_state_dict(converted,strict=True)
    class Client(fl.client.NumPyClient):
        def get_parameters(self,config):return get_params()
        def fit(self,parameters,config):
            nonlocal iterator,step,total_samples,round_num,previous_ready_step,opt,scaler,epoch
            if previous_ready_step is not None:
                emit('comm_complete',detail='next_round_received_after_previous_weights_sent')
            round_num=int(config.get('round',round_num+1))
            round_samples=0
            set_params(parameters)
            emit('model_apply',detail=f'flower_round={round_num}')
            # E1_b Flower rebuilds the SGD state on every FedAvg round.
            current_lr=lr_at(min(epoch,a.epochs-1),a)
            opt=torch.optim.SGD(model.parameters(),lr=current_lr,momentum=a.momentum,
                                weight_decay=a.weight_decay)
            scaler=torch.amp.GradScaler('cuda',enabled=amp)
            model.train()
            for _ in range(a.steps_per_round):
                wait_start=time.perf_counter()
                try:x,y=next(iterator)
                except StopIteration:
                    epoch+=1
                    sampler.set_epoch(epoch)
                    iterator=iter(loader)
                    x,y=next(iterator)
                t0=time.perf_counter();batch_wait=t0-wait_start
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                sync_cuda(torch);t1=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    logits=model(x);loss=criterion(logits,y)
                sync_cuda(torch);t2=time.perf_counter()
                scaler.scale(loss).backward();sync_cuda(torch);t3=time.perf_counter()
                scaler.step(opt);scaler.update();sync_cuda(torch);t4=time.perf_counter()
                step+=1;total_samples+=len(y);round_samples+=len(y)
                emit('local_optimizer_step')
                now=time.time()
                prog.writerow(dict(experiment='E2',host=socket.gethostname(),rank=rank,
                    local_rank=0,epoch=round_num,step=step,unix_s=now,unix_time_s=now,
                    elapsed_s=now-a.start,loss=float(loss.detach()),lr=current_lr,
                    local_samples=total_samples,global_samples_approx=4*total_samples,
                    batch_samples=len(y),step_s=t4-t0,batch_wait_s=batch_wait,
                    h2d_s=t1-t0,forward_s=t2-t1,backward_with_comm_s=t3-t2,
                    optimizer_s=t4-t3,other_s=0,collectives=0,
                    collective_bytes_logical=0,collective_sum_host_s=0,
                    collective_span_host_s=0,collective_max_host_s=0,
                    collective_failures=0,gpu_memory_allocated_mb=torch.cuda.memory_allocated(device)/(1024**2),
                    samples_per_s=len(y)/max(t4-t0,1e-9),step_wall_s=t4-wait_start))
            arrays=get_params()
            nbytes=sum(x.nbytes for x in arrays)
            emit('comm_ready',bytes=nbytes,detail='client_weights_ready_for_send')
            previous_ready_step=step
            return arrays,round_samples,{'steps':step,'round':round_num,'worker_rank':rank,'local_samples':total_samples}
        def evaluate(self,parameters,config):
            # Evaluate the SAME global FedAvg parameters on all four ranks.
            set_params(parameters)
            acc,loss,n,seconds=evaluate_local(model,val_loader,device,torch,amp=amp)
            now=time.time()
            vw.writerow(dict(experiment='E2',method=f'fl{a.steps_per_round}',
                rank=rank,epoch=int(config.get('round',round_num)),step=step,
                unix_time_s=now,elapsed_s=now-a.start,
                val_accuracy=acc,val_loss=loss,val_samples=n,val_s=seconds,
                local_samples=total_samples,global_samples_approx=total_samples*4))
            emit('global_model_evaluated',detail=f'round={config.get("round",round_num)};accuracy={acc:.5f}')
            return loss,n,{'worker_rank':rank,'accuracy':acc,'eval_seconds':seconds,'step':step}
    emit('worker_ready')
    try:
        fl.client.start_client(server_address=a.server,client=Client().to_client(),
                               grpc_max_message_length=2147483647)
    finally:
        if resources:resources.close()
        vf.close();fp.close();fe.close()
if __name__=='__main__':main()
