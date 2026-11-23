#!/usr/bin/env python3
"""E1_b Flower FedAvg (4 GPUs, one client process per GPU; separate server).

Pinned compatibility target: Flower 1.24.0 legacy gRPC API.
Pure PyTorch model/optimizer, Flower handles communication and FedAvg aggregation.
Uses E1_a_new's shared e1a_reference.py for identical CIFAR-100 data/metric schema.
"""
import argparse
import csv
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
except ImportError as exc:
    raise SystemExit('Flower missing. Install in the shared environment: python -m pip install flwr==1.24.0') from exc
try:
    import e1a_reference as ref
except ImportError as exc:
    raise SystemExit('Missing e1a_reference.py; run E1_b/install.sh once to copy E1_a_new files') from exc

MSG_FIELDS=['rank','host','step','epoch','unix_time_s','elapsed_s','event','src_rank','dst_rank',
            'src_ip','dst_ip','payload_bytes_logical','duration_host_s','local_version',
            'received_version','status','method','measurement_scope']
WORKER_VAL_FIELDS=['experiment','method','rank','epoch','step','unix_time_s','elapsed_s',
                   'val_loss','val_accuracy','val_samples','val_s','local_samples','global_samples_approx']
AGG_VAL_FIELDS=['experiment','method','epoch','step','elapsed_s','unix_time_s',
                'mean_accuracy','min_accuracy','max_accuracy','std_accuracy',
                'mean_loss','min_loss','max_loss','std_loss','val_samples_per_worker',
                'world_size','mean_accuracy_target_reached','all_workers_target_reached']
ROUND_FIELDS=['round','elapsed_s','fit_workers','eval_workers','train_samples',
              'model_payload_bytes_per_client','logical_upload_bytes','logical_download_bytes',
              'mean_client_train_seconds','aggregate_host_seconds','mean_accuracy','global_model_accuracy',
              'mean_loss','global_model_loss','round_failure_count']


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['server','client'],required=True)
    p.add_argument('--server-address',required=True,help='server IPv4:port reachable by all four workers')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=4)
    p.add_argument('--gpu-index',type=int,default=0,help='physical GPU index for resource sampler; CUDA_VISIBLE_DEVICES masks it to cuda:0')
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='../data')
    p.add_argument('--dataset',choices=['cifar100','fake'],default='cifar100')
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--local-steps',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=100,help='reference LR schedule horizon in epochs')
    p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--min-lr',type=float,default=.0001)
    p.add_argument('--warmup-epochs',type=int,default=5)
    p.add_argument('--momentum',type=float,default=.9)
    p.add_argument('--weight-decay',type=float,default=.0005)
    p.add_argument('--label-smoothing',type=float,default=0.)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--val-size',type=int,default=5000)
    p.add_argument('--validation-source',choices=['train-holdout','official-test'],default='train-holdout')
    p.add_argument('--eval-every-epochs',type=int,default=1)
    p.add_argument('--log-every-steps',type=int,default=10)
    p.add_argument('--target-accuracy',type=float,default=.7)
    p.add_argument('--max-seconds',type=float,default=3600,help='advisory; fixed Flower round count is termination condition')
    p.add_argument('--sample-period-s',type=float,default=1.)
    p.add_argument('--gpu-idle-threshold',type=float,default=20.)
    p.add_argument('--net-iface',default='br0')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--train-subset',type=int,default=0)
    p.add_argument('--val-subset',type=int,default=0)
    p.add_argument('--cpu-smoke',action='store_true')
    a=p.parse_args()
    if min(a.world_size,a.rounds,a.local_steps,a.batch_size,a.epochs,a.log_every_steps,a.eval_every_epochs)<1:
        p.error('counts must be positive')
    if not 0<=a.warmup_epochs<a.epochs:p.error('invalid LR warmup')
    if not 0<=a.rank<a.world_size:p.error('rank out of range')
    return a


def make_model():
    model=models.resnet18(weights=None,num_classes=100)
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    return model


def state_arrays(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def load_arrays(model, arrays):
    keys=list(model.state_dict())
    if len(keys)!=len(arrays):raise ValueError('Model parameter/buffer array count mismatch')
    vals={k:torch.from_numpy(np.asarray(a).copy()) for k,a in zip(keys,arrays)}
    model.load_state_dict(vals,strict=True)


def bytes_of(arrays):return int(sum(a.nbytes for a in arrays))


def open_writer(path,fields):
    f=Path(path).open('w',newline='',buffering=1)
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
    return f,w


class Client(fl.client.NumPyClient):
    def __init__(self,a):
        self.a=a;self.rank=a.rank;self.world=a.world_size
        self.host=socket.gethostname();self.ip=os.environ.get('E1_WORKER_IP',socket.gethostbyname(self.host))
        self.out=Path(a.output);self.out.mkdir(parents=True,exist_ok=True)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if not a.cpu_smoke:
            # Each launched client is assigned exactly one GPU by CUDA_VISIBLE_DEVICES.
            torch.cuda.set_device(0)
        self.device=torch.device('cpu' if a.cpu_smoke else 'cuda:0')
        if self.device.type=='cuda':
            torch.cuda.manual_seed_all(a.seed);torch.backends.cudnn.benchmark=True
        self.model=make_model().to(self.device)
        # Dataset and validation split are the *same* as the E1_a_new reference.
        self.trainloader,self.valloader,self.sampler,self.train_n,self.val_n=ref.data_loaders(a,a.rank,a.world_size)
        if self.valloader is None:
            _,self.valloader,_,_,nval=ref.data_loaders(a,0,a.world_size)
            if nval!=self.val_n:raise RuntimeError('Different validation split on nonzero rank')
        if self.valloader is None:raise RuntimeError('Validation loader missing')
        self.loader_epoch=0;self.sampler.set_epoch(0);self.iterator=iter(self.trainloader)
        self.step=0;self.samples=0;self.start=time.perf_counter();self.start_unix=time.time()
        self.loss_fn=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
        self.amp=(self.device.type=='cuda' and not a.no_amp)
        self.optimizer=None
        self.sf,self.sw=open_writer(self.out/f'steps_rank{a.rank}.csv',ref.STEP_FIELDS)
        self.cf,self.cw=open_writer(self.out/f'collectives_rank{a.rank}.csv',ref.COLL_FIELDS)
        self.yf,self.yw=open_writer(self.out/f'sync_events_rank{a.rank}.csv',ref.SYNC_FIELDS)
        self.mf,self.mw=open_writer(self.out/f'messages_rank{a.rank}.csv',MSG_FIELDS)
        self.vf,self.vw=open_writer(self.out/f'validation_rank{a.rank}.csv',WORKER_VAL_FIELDS)
        self.resource=None;self.tcp=None;self.tcp_log=None;self.tcp_stop=None
        if self.device.type=='cuda' and a.sample_period_s>0:
            self.resource=ref.SamplerThread(self.out,a.rank,a.gpu_index,a.net_iface,a.sample_period_s,time.time())
            self.resource.start()
        # One rootless network watcher per physical host, run only from local GPU index 0.
        if a.gpu_index==0 and os.environ.get('E1_TCP_MEASURE','0')=='1':
            script=Path(__file__).with_name('socket_counter_watch.py')
            if script.exists():
                self.tcp_stop=self.out/f'monitor_stop_{self.host.split(".")[0]}'
                self.tcp_log=(self.out/f'monitor_launch_{self.host.split(".")[0]}.log').open('w',buffering=1)
                env=os.environ.copy();env['NET_IFACE']=a.net_iface
                self.tcp=subprocess.Popen([sys.executable,str(script),str(self.out),
                    os.environ.get('E1_TCP_PERIOD','0.1'),os.environ.get('E1_TCP_DURATION','3600'),
                    '--stop-file',str(self.tcp_stop)],stdout=self.tcp_log,stderr=subprocess.STDOUT,env=env)
        (self.out/f'worker_{a.rank}_meta.json').write_text(json.dumps({
            'rank':a.rank,'node':self.host,'ip':self.ip,'pid':os.getpid(),
            'device':str(self.device),'data_samples':self.train_n,
            'dataset':'E1_a_new stratified holdout, local distributed partition'},indent=2))
        self.comm_bytes=0;self.comm_events=0;self.comm_sum=0.;self.comm_max=0.

    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def elapsed(self):return time.perf_counter()-self.start

    def message(self,event,round_num,payload,duration,source,destination,status='ok'):
        now=time.time()
        self.mw.writerow(dict(rank=self.rank,host=self.host,step=self.step,epoch=self.loader_epoch+1,
          unix_time_s=round(now,6),elapsed_s=round(self.elapsed(),6),event=event,
          src_rank=source,dst_rank=destination,src_ip=self.ip if source==self.rank else '',
          dst_ip=self.ip if destination==self.rank else '',payload_bytes_logical=payload,
          duration_host_s=round(duration,7),local_version=round_num,received_version=round_num,
          status=status,method='flower_fedavg',measurement_scope='serialized model tensor payload, excludes gRPC overhead'))
        self.cw.writerow(dict(rank=self.rank,step=self.step,bucket_index=0,bytes_logical=payload,
            start_unix_s=round(now-duration,6),end_unix_s=round(now,6),duration_host_s=round(duration,7),
            status=status,src_node=self.host,src_ip=self.ip,collective_membership=event,
            destination_rank=destination,destination_ip='server' if destination=='server' else self.ip,
            measurement_scope='application tensor payload, not observed TCP bytes'))
        self.comm_bytes+=payload;self.comm_events+=1;self.comm_sum+=duration
        self.comm_max=max(self.comm_max,duration)

    def get_parameters(self, config):
        return state_arrays(self.model)

    def fit(self, parameters, config):
        round_num=int(config.get('round',0))
        begin=time.perf_counter()
        load_arrays(self.model,parameters)
        self.message('fedavg_model_download',round_num,bytes_of(parameters),time.perf_counter()-begin,'server',self.rank)
        # Reset local SGD momentum after each FedAvg aggregation (explicit algorithm choice).
        lr=ref.lr_at(min(self.loader_epoch,self.a.epochs-1),self.a)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=lr,momentum=self.a.momentum,
                                       weight_decay=self.a.weight_decay)
        scaler=torch.amp.GradScaler('cuda',enabled=self.amp)
        self.model.train();local_count=0;loss_sum=0.;batch_count=0
        fit_begin=time.perf_counter()
        for _ in range(self.a.local_steps):
            wait_start=time.perf_counter()
            try:images,labels=next(self.iterator)
            except StopIteration:
                self.loader_epoch+=1;self.sampler.set_epoch(self.loader_epoch)
                self.iterator=iter(self.trainloader)
                images,labels=next(self.iterator)
            t0=time.perf_counter();batch_wait=t0-wait_start
            images=images.to(self.device,non_blocking=(self.device.type=='cuda'))
            labels=labels.to(self.device,non_blocking=(self.device.type=='cuda'))
            self.synchronize();t1=time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.amp):
                logits=self.model(images);loss=self.loss_fn(logits,labels)
            self.synchronize();t2=time.perf_counter()
            scaler.scale(loss).backward();self.synchronize();t3=time.perf_counter()
            scaler.step(self.optimizer);scaler.update();self.synchronize();t4=time.perf_counter()
            self.step+=1;bs=labels.size(0);self.samples+=bs;local_count+=bs;batch_count+=1
            loss_sum+=loss.item()*bs
            if self.step%self.a.log_every_steps==0:
                self.sw.writerow(dict(experiment='E1_b',host=self.host,rank=self.rank,local_rank=self.a.gpu_index,
                    epoch=self.loader_epoch+1,step=self.step,unix_time_s=round(time.time(),6),
                    elapsed_s=round(self.elapsed(),6),loss=round(float(loss.item()),7),lr=round(lr,9),
                    local_samples=self.samples,global_samples_approx=self.samples*self.world,
                    batch_samples=bs,step_s=round(t4-t0,7),batch_wait_s=round(batch_wait,7),
                    h2d_s=round(t1-t0,7),forward_s=round(t2-t1,7),backward_with_comm_s=round(t3-t2,7),
                    optimizer_s=round(t4-t3,7),other_s=0.,
                    collectives=self.comm_events,collective_bytes_logical=self.comm_bytes,
                    collective_sum_host_s=round(self.comm_sum,7),collective_span_host_s=round(self.comm_sum,7),
                    collective_max_host_s=round(self.comm_max,7),collective_failures=0,
                    gpu_memory_allocated_mb=round(torch.cuda.memory_allocated(self.device)/1048576,2) if self.device.type=='cuda' else 0.,
                    samples_per_s=round(bs/max(t4-t0,1e-9),3),step_wall_s=round(t4-wait_start,7)))
                self.comm_events=0;self.comm_bytes=0;self.comm_sum=0.;self.comm_max=0.
        weights=state_arrays(self.model)
        self.message('fedavg_model_upload',round_num,bytes_of(weights),0.,self.rank,'server',status='submitted_to_flower')
        # Flower performs actual gRPC transfer after this callback returns. The callback
        # has no access to precise transport completion latency; use TCP monitor for that.
        return weights,local_count,{'worker_rank':self.rank,'local_samples':self.samples,
               'local_steps':batch_count,'client_train_seconds':time.perf_counter()-fit_begin,
               'train_loss':loss_sum/max(1,local_count)}

    def evaluate(self, parameters,config):
        round_num=int(config.get('round',0))
        begin=time.perf_counter();load_arrays(self.model,parameters)
        self.message('fedavg_global_eval_download',round_num,bytes_of(parameters),time.perf_counter()-begin,'server',self.rank)
        # Unlike asynchronous gossip, all Flower clients evaluate the SAME global model.
        self.model.eval();self.synchronize();start=time.perf_counter()
        count=correct=0;loss_total=0.
        with torch.inference_mode():
            for images,labels in self.valloader:
                images=images.to(self.device,non_blocking=(self.device.type=='cuda'))
                labels=labels.to(self.device,non_blocking=(self.device.type=='cuda'))
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.amp):
                    logits=self.model(images);loss=self.loss_fn(logits,labels)
                bs=labels.numel();count+=bs;correct+=int((logits.argmax(1)==labels).sum().item())
                loss_total+=loss.item()*bs
        self.synchronize();dt=time.perf_counter()-start
        accuracy=correct/max(1,count);val_loss=loss_total/max(1,count)
        self.vw.writerow(dict(experiment='E1_b',method='flower_fedavg',rank=self.rank,
            epoch=round_num,step=self.step,unix_time_s=round(time.time(),6),elapsed_s=round(self.elapsed(),6),
            val_loss=round(val_loss,7),val_accuracy=round(accuracy,7),val_samples=count,
            val_s=round(dt,7),local_samples=self.samples,global_samples_approx=self.samples*self.world))
        return val_loss,count,{'worker_rank':self.rank,'accuracy':accuracy,'eval_seconds':dt,'val_samples':count,'step':self.step}

    def close(self):
        if self.tcp is not None:
            try:
                self.tcp_stop.touch();self.tcp.wait(timeout=8)
            except Exception:
                self.tcp.terminate()
        if self.tcp_log:self.tcp_log.close()
        if self.resource:self.resource.close()
        for f in (self.sf,self.cf,self.yf,self.mf,self.vf):f.close()


class FedAvgMeter(fl.server.strategy.FedAvg):
    def __init__(self,a):
        self.a=a;self.out=Path(a.output);self.out.mkdir(parents=True,exist_ok=True)
        self.start=time.perf_counter();self.best=0.;self.first_target=None
        self.fit_metrics_by_round={};self.eval_metrics_by_round={}
        self.latest_parameters=None;self.global_arrays=None
        torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
        initial=state_arrays(make_model())
        self.model_bytes=bytes_of(initial)
        self.summary_writer=open_writer(self.out/'rounds.csv',ROUND_FIELDS)
        super().__init__(fraction_fit=1.0,fraction_evaluate=1.0,
          min_fit_clients=a.world_size,min_evaluate_clients=a.world_size,
          min_available_clients=a.world_size,accept_failures=False,
          initial_parameters=ndarrays_to_parameters(initial),
          on_fit_config_fn=lambda r:{'round':r},
          on_evaluate_config_fn=lambda r:{'round':r})
        self.global_arrays=initial
        self.fit_rows=[]

    def aggregate_fit(self,server_round,results,failures):
        t0=time.perf_counter()
        if failures or len(results)!=self.a.world_size:
            raise RuntimeError(f'Round {server_round}: missing/failed workers, results={len(results)}, failures={len(failures)}')
        params,agg=super().aggregate_fit(server_round,results,failures)
        if params is None:raise RuntimeError('Flower returned no aggregated model')
        self.latest_parameters=params;self.global_arrays=parameters_to_ndarrays(params)
        duration=time.perf_counter()-t0
        ordered=sorted(results,key=lambda item:int(item[1].metrics['worker_rank']))
        self.fit_metrics_by_round[server_round]=dict(
            fit_workers=len(results),train_samples=sum(int(res.num_examples) for _,res in results),
            mean_client_train_seconds=float(np.mean([float(res.metrics['client_train_seconds']) for _,res in results])),
            aggregate_host_seconds=duration)
        for _,res in ordered:
            self.fit_rows.append({'round':server_round,'rank':int(res.metrics['worker_rank']),
               'train_samples':int(res.num_examples),'train_seconds':float(res.metrics['client_train_seconds']),
               'train_loss':float(res.metrics['train_loss'])})
        # Rank0-style legacy checkpoint = global FedAvg model, not any client snapshot.
        model=make_model();load_arrays(model,self.global_arrays)
        tmp=self.out/'checkpoint_latest.part'
        torch.save({'model':model.state_dict(),'method':'flower_fedavg','round':server_round,
                    'global_model':True,'config':vars(self.a)},tmp)
        os.replace(tmp,self.out/'checkpoint_latest.pt')
        return params,agg

    def aggregate_evaluate(self,server_round,results,failures):
        if failures or len(results)!=self.a.world_size:
            raise RuntimeError(f'Round {server_round} validation missing workers')
        loss,agg=super().aggregate_evaluate(server_round,results,failures)
        sorted_results=sorted(results,key=lambda z:int(z[1].metrics['worker_rank']))
        acc=np.array([float(res.metrics['accuracy']) for _,res in sorted_results]);
        losses=np.array([float(res.loss) for _,res in sorted_results]);n=np.array([int(res.num_examples) for _,res in sorted_results]);
        if len(set(n))!=1:raise RuntimeError('Workers evaluated unequal holdout sizes')
        timestamp=time.perf_counter()-self.start
        mean=float(np.mean(acc));minimum=float(np.min(acc));maximum=float(np.max(acc))
        if self.first_target is None and self.a.target_accuracy>0 and mean>=self.a.target_accuracy:
            self.first_target=timestamp
        self.eval_metrics_by_round[server_round]=(mean,float(np.mean(losses)))
        # For FedAvg these are replicated evaluations of the SAME global model.
        vf,vw=open_writer(self.out/'validation_aggregate.csv',AGG_VAL_FIELDS) if server_round==1 else (None,None)
        if vf is not None: self.agg_vf,self.agg_vw=vf,vw
        self.agg_vw.writerow(dict(experiment='E1_b',method='flower_fedavg',epoch=server_round,step=server_round,
            elapsed_s=round(timestamp,6),unix_time_s=time.time(),mean_accuracy=mean,
            min_accuracy=minimum,max_accuracy=maximum,std_accuracy=float(np.std(acc)),
            mean_loss=float(np.mean(losses)),min_loss=float(np.min(losses)),
            max_loss=float(np.max(losses)),std_loss=float(np.std(losses)),
            val_samples_per_worker=int(n[0]),world_size=self.a.world_size,
            mean_accuracy_target_reached=int(self.a.target_accuracy>0 and mean>=self.a.target_accuracy),
            all_workers_target_reached=int(self.a.target_accuracy>0 and minimum>=self.a.target_accuracy)))
        # E1_a_new-compatible validation.csv: rank 0 evaluates global model.
        if server_round==1: self.legacy_f,self.legacy_w=open_writer(self.out/'validation.csv',ref.VAL_FIELDS)
        r0=next(res for _,res in results if int(res.metrics['worker_rank'])==0)
        seen=sum(int(row['train_samples']) for row in self.fit_rows)
        self.legacy_w.writerow(dict(experiment='E1_b',epoch=server_round,step=server_round,
            global_samples_approx=seen,unix_time_s=time.time(),elapsed_s=round(timestamp,6),
            val_loss=float(r0.loss),val_accuracy=float(r0.metrics['accuracy']),val_samples=int(r0.num_examples),
            val_s=float(r0.metrics['eval_seconds']),lr=self.a.lr,
            best_val_accuracy=max(x[0] for x in self.eval_metrics_by_round.values()),
            cumulative_samples_per_s=seen/max(timestamp,1e-9)))
        fit=self.fit_metrics_by_round[server_round]
        self.summary_writer[1].writerow(dict(round=server_round,elapsed_s=round(timestamp,6),
            **fit,eval_workers=len(results),model_payload_bytes_per_client=self.model_bytes,
            logical_upload_bytes=self.model_bytes*self.a.world_size,
            logical_download_bytes=self.model_bytes*self.a.world_size*2,
            mean_accuracy=mean,global_model_accuracy=mean,
            mean_loss=float(np.mean(losses)),global_model_loss=float(np.mean(losses)),round_failure_count=0))
        print('ROUND',json.dumps(dict(round=server_round,global_accuracy=mean,elapsed_s=timestamp)),flush=True)
        return loss,{'accuracy':mean}

    def save_summary(self):
        out=self.out
        if hasattr(self,'agg_vf'):self.agg_vf.close()
        if hasattr(self,'legacy_f'):self.legacy_f.close()
        self.summary_writer[0].close()
        if self.fit_rows:
            with (out/'fit_clients.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(self.fit_rows[0]));w.writeheader();w.writerows(self.fit_rows)
        vals=list(self.eval_metrics_by_round.items())
        summary=dict(experiment='E1_b',method='flower_fedavg',status='ok' if len(vals)==self.a.rounds else 'incomplete',
            world_size=self.a.world_size,completed_rounds=len(vals),requested_rounds=self.a.rounds,
            wall_training_s=round(time.perf_counter()-self.start,4),
            best_validation_accuracy=max((v[0] for _,v in vals),default=None),
            target_accuracy=self.a.target_accuracy,target_first_s=self.first_target,
            global_model_validation='FedAvg global model, evaluated independently by all four clients',
            training_stop='fixed Flower round count; max_seconds is advisory only',
            global_batch_size=self.a.world_size*self.a.batch_size)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))


def main():
    a=parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.role=='server':
        print('Flower server',fl.__version__,'address',a.server_address,flush=True)
        (out/'manifest.json').write_text(json.dumps(dict(experiment='E1_b',method='flower_fedavg',
            flower=fl.__version__,torch=torch.__version__,settings=vars(a),
            note='Legacy Flower gRPC for fixed physical Grid5000 worker mapping; all four participate every round',
            training_budget='fixed round count; do not claim hard 3600-s budget'),indent=2))
        strategy=FedAvgMeter(a)
        try:
            fl.server.start_server(server_address='0.0.0.0:'+a.server_address.rsplit(':',1)[1],
                config=fl.server.ServerConfig(num_rounds=a.rounds),strategy=strategy)
        finally:strategy.save_summary()
    else:
        client=Client(a)
        try:
            fl.client.start_client(server_address=a.server_address,client=client.to_client())
        finally:client.close()

if __name__=='__main__':main()
