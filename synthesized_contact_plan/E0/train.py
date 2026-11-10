#!/usr/bin/env python3
"""E0 v2: one-GPU CIFAR-100 ResNet18, warmup + cosine SGD, AMP, checkpoint/resume.

Research note: >70% is a target, never guaranteed. Default accuracy uses a fixed
stratified validation holdout from the 50,000 training images; test is untouched.
"""
import argparse, csv, json, math, os, random, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

STEP_COLUMNS=['experiment','epoch','step','samples_seen','elapsed_s','loss','lr','batch_samples','step_s','data_s','forward_s','backward_s','optimizer_s','sync_s','samples_per_s','gpu_memory_allocated_mb']
VAL_COLUMNS=['experiment','epoch','step','samples_seen','elapsed_s','val_loss','val_accuracy','val_samples','val_s','lr','best_val_accuracy','train_samples_per_s_running']

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--data-root',default='./data')
    p.add_argument('--download',action='store_true')
    p.add_argument('--dataset',choices=['cifar100','fake'],default='cifar100')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.1)
    p.add_argument('--min-lr',type=float,default=0.0001)
    p.add_argument('--warmup-epochs',type=int,default=5)
    p.add_argument('--momentum',type=float,default=0.9)
    p.add_argument('--weight-decay',type=float,default=0.0005)
    p.add_argument('--label-smoothing',type=float,default=0.0)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--eval-every-epochs',type=int,default=1)
    p.add_argument('--log-every-steps',type=int,default=10)
    p.add_argument('--max-seconds',type=float,default=0,help='0 = no cutoff. Time measured from first batch.')
    p.add_argument('--target-accuracy',type=float,default=0.70,help='log target crossing (fraction), do not stop automatically')
    p.add_argument('--val-size',type=int,default=5000,help='fixed, stratified holdout from train set')
    p.add_argument('--validation-source',choices=['train-holdout','official-test'],default='train-holdout')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--resume',default='',help='path to checkpoint_latest.pt; must use same run dir and hyperparams')
    p.add_argument('--gpu-sample-s',type=int,default=1)
    p.add_argument('--train-subset',type=int,default=0,help='smoke-test only')
    p.add_argument('--val-subset',type=int,default=0,help='smoke-test only')
    a=p.parse_args()
    if any(x<1 for x in [a.epochs,a.batch_size,a.eval_every_epochs,a.log_every_steps]):p.error('positive epoch/batch/evaluation/log counts required')
    if a.warmup_epochs>=a.epochs or a.warmup_epochs<0:p.error('warmup epochs must be in [0,epochs)')
    if not 0<=a.target_accuracy<=1:p.error('target accuracy fraction must be in [0,1]')
    return a

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def lr_at_epoch(epoch_idx,a):
    # Epoch 0..warmup-1 warm-up to base LR; then cosine down to min LR.
    if a.warmup_epochs and epoch_idx<a.warmup_epochs:
        return a.lr*(epoch_idx+1)/a.warmup_epochs
    progress=(epoch_idx-a.warmup_epochs)/max(1,a.epochs-a.warmup_epochs-1)
    progress=min(1.0,max(0.0,progress))
    return a.min_lr+0.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*progress))

def prepare_data(a,device):
    mean=(0.5071,0.4867,0.4408);std=(0.2675,0.2565,0.2761)
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
    eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
    if a.dataset=='fake':
        print('WARNING: FakeData smoke test ONLY. Accuracy has no scientific meaning.',flush=True)
        raw_train=datasets.FakeData(size=512,image_size=(3,32,32),num_classes=100,transform=train_tf)
        raw_val=datasets.FakeData(size=128,image_size=(3,32,32),num_classes=100,transform=eval_tf,random_offset=50000)
        tr_idxs=list(range(len(raw_train)));val_idxs=list(range(len(raw_val)))
        train=Subset(raw_train,tr_idxs);val=Subset(raw_val,val_idxs)
    else:
        raw_train=datasets.CIFAR100(a.data_root,train=True,download=a.download,transform=train_tf)
        raw_eval_train=datasets.CIFAR100(a.data_root,train=True,download=False,transform=eval_tf)
        if a.validation_source=='train-holdout':
            if not 0<a.val_size<len(raw_train):raise ValueError('val-size must be between 1 and 49999')
            # Stratified deterministic split: 100 classes * 500 images each
            labels=np.asarray(raw_train.targets)
            rng=np.random.default_rng(a.seed)
            by_class=[rng.permutation(np.where(labels==c)[0]) for c in range(100)]
            q,r=divmod(a.val_size,100)
            val_idxs=np.concatenate([v[:q+(i<r)] for i,v in enumerate(by_class)]).tolist()
            tr_idxs=np.concatenate([v[q+(i<r):] for i,v in enumerate(by_class)]).tolist()
            train=Subset(raw_train,tr_idxs);val=Subset(raw_eval_train,val_idxs)
        else:
            val=datasets.CIFAR100(a.data_root,train=False,download=a.download,transform=eval_tf)
            train=raw_train;tr_idxs=list(range(len(train)));val_idxs=list(range(len(val)))
    if a.train_subset:train=Subset(train,list(range(min(a.train_subset,len(train)))))
    if a.val_subset:val=Subset(val,list(range(min(a.val_subset,len(val)))))
    train_generator=torch.Generator().manual_seed(a.seed)
    kw=dict(num_workers=a.num_workers,pin_memory=device.type=='cuda',persistent_workers=a.num_workers>0)
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=train_generator,**kw)
    val_loader=DataLoader(val,batch_size=max(128,a.batch_size),shuffle=False,**kw)
    return train_loader,val_loader,train_generator,dict(train_n=len(train),val_n=len(val),validation_source=a.validation_source,split_seed=a.seed)

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()

def save_atomic(obj,path):
    tmp=path.with_suffix('.part')
    torch.save(obj,tmp);os.replace(tmp,path)

def main():
    a=parser(); out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if a.resume and not Path(a.resume).is_file():raise FileNotFoundError(a.resume)
    if a.resume and Path(a.resume).resolve().parent != out:raise ValueError('Resume must use checkpoint from the same output directory')
    seed_everything(a.seed)
    device=torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda:0')
    amp=(device.type=='cuda') and not a.no_amp
    trainloader,valloader,generator,dinfo=prepare_data(a,device)
    model=models.resnet18(weights=None,num_classes=100)
    model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity()
    model.to(device)
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay,nesterov=False)
    loss_fn=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    start_epoch=0;step=0;samples_seen=0;best_acc=-1.0;elapsed_before=0.0;target_first=None
    resume_obj=None
    if a.resume:
        resume_obj=torch.load(a.resume,map_location='cpu',weights_only=False)
        if resume_obj.get('epoch_completed',0)<0:raise ValueError('Invalid checkpoint epoch')
        # Avoid resuming into an incompatible experiment recipe.
        old=resume_obj.get('config',{})
        keys=['epochs','batch_size','seed','lr','min_lr','warmup_epochs','momentum','weight_decay','label_smoothing','validation_source','val_size','dataset']
        mismatch=[k for k in keys if old.get(k)!=getattr(a,k)]
        if mismatch:raise ValueError('Incompatible checkpoint settings: '+', '.join(mismatch))
        model.load_state_dict(resume_obj['model']);opt.load_state_dict(resume_obj['optimizer'])
        scaler.load_state_dict(resume_obj['scaler']);start_epoch=resume_obj['epoch_completed']
        step=resume_obj['step'];samples_seen=resume_obj['samples_seen']
        best_acc=resume_obj['best_val_accuracy'];elapsed_before=resume_obj['elapsed_s']
        target_first=resume_obj.get('target_first_s')
        random.setstate(resume_obj['random_state']);np.random.set_state(resume_obj['numpy_state'])
        torch.set_rng_state(resume_obj['torch_rng'])
        if torch.cuda.is_available() and resume_obj.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(resume_obj['cuda_rng'])
        generator.set_state(resume_obj['data_generator_state'])
        print(f'RESUMED after epoch={start_epoch} step={step} elapsed={elapsed_before:.1f}s',flush=True)
    meta={'experiment':'E0_v2','hostname':socket.gethostname(),'start_utc':datetime.now(timezone.utc).isoformat(),
          'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,
          'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','device':str(device),
          'amp':amp,'arguments':vars(a),'dataset':dinfo,
          'notes':['Fresh initialization unless --resume is set.','Validation holdout is disjoint from training when train-holdout is used.',
                   'Target 70% is a success criterion, not a guarantee.','Epoch-end cosine learning rate updated at epoch boundaries.',
                   'No DDP, no communication; sync_s=0 is a defined placeholder.',
                   'Checkpoint latest resumes only after completed epochs; interrupted incomplete epoch is not saved.']}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,default=str))
    step_path=out/'steps.csv';val_path=out/'validation.csv'
    def openlog(path,fields):
        exists=bool(a.resume and path.exists() and path.stat().st_size>0)
        f=path.open('a' if exists else 'w',newline='',buffering=1);writer=csv.DictWriter(f,fieldnames=fields)
        if not exists:writer.writeheader()
        return f,writer
    sf,sw=openlog(step_path,STEP_COLUMNS);vf,vw=openlog(val_path,VAL_COLUMNS)
    gpuout=None;gpuproc=None
    if device.type=='cuda' and a.gpu_sample_s>0:
        try:
            gpuout=(out/'gpu.csv').open('a' if a.resume else 'w',buffering=1)
            gpuproc=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv,noheader,nounits','-l',str(a.gpu_sample_s)],stdout=gpuout,stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            if gpuout:gpuout.close()
            gpuout=None
    tstart=time.perf_counter();stop_reason='epochs_completed';last_val=None;train_loss_sum=0.0;total_train_count=0
    def elapsed():return elapsed_before+(time.perf_counter()-tstart)
    best_checkpoint_needed=[False]
    def checkpoint(epoch_complete,val_acc):
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
            'config':vars(a),'epoch_completed':epoch_complete,'step':step,'samples_seen':samples_seen,
            'best_val_accuracy':best_acc,'elapsed_s':elapsed(),'target_first_s':target_first,
            'random_state':random.getstate(),'numpy_state':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
            'data_generator_state':generator.get_state(),'last_val_accuracy':val_acc}
        save_atomic(ck,out/'checkpoint_latest.pt')
        if val_acc>=best_acc and val_acc>=0 and best_checkpoint_needed[0]:
            save_atomic(ck,out/'checkpoint_best.pt')
            best_checkpoint_needed[0]=False
    def evaluate(epoch):
        nonlocal best_acc,last_val,target_first
        model.eval();sync(device);t0=time.perf_counter()
        count=0;correct=0;loss_sum=0.
        with torch.inference_mode():
            for images,targets in valloader:
                images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    pred=model(images);loss=loss_fn(pred,targets)
                bs=targets.size(0);count+=bs;correct+=int((pred.argmax(1)==targets).sum().item());loss_sum+=float(loss.item())*bs
        sync(device);validation_s=time.perf_counter()-t0;now=elapsed();acc=correct/max(1,count)
        if acc>best_acc:
            best_acc=acc
            best_checkpoint_needed[0]=True
        if target_first is None and a.target_accuracy>0 and acc>=a.target_accuracy:target_first=now
        last_val={'experiment':'E0_v2','epoch':epoch,'step':step,'samples_seen':samples_seen,
                  'elapsed_s':round(now,4),'val_loss':round(loss_sum/max(1,count),6),
                  'val_accuracy':round(acc,6),'val_samples':count,'val_s':round(validation_s,4),
                  'lr':round(opt.param_groups[0]['lr'],9),'best_val_accuracy':round(best_acc,6),
                  'train_samples_per_s_running':round(samples_seen/max(1e-9,now),2)}
        vw.writerow(last_val)
        print('VALIDATION',json.dumps(last_val),flush=True)
        model.train()
        return acc
    try:
        for epoch_index in range(start_epoch,a.epochs):
            # Schedule is tied to number of epochs (progress) rather than elapsed wall time.
            current_lr=lr_at_epoch(epoch_index,a)
            for group in opt.param_groups:group['lr']=current_lr
            model.train();loss_sum_epoch=0.;count_epoch=0
            for batch_i,(images,targets) in enumerate(trainloader):
                # Budget is checked at batch boundaries. If it expires mid-epoch,
                # the run saves logs but only last completed epoch is resumable.
                if a.max_seconds and elapsed()>=a.max_seconds:
                    stop_reason='time_budget';break
                t0=time.perf_counter()
                images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
                sync(device);t1=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    pred=model(images);loss=loss_fn(pred,targets)
                sync(device);t2=time.perf_counter()
                scaler.scale(loss).backward();sync(device);t3=time.perf_counter()
                scaler.step(opt);scaler.update();sync(device);t4=time.perf_counter()
                bs=targets.size(0);batch_loss=float(loss.item());step+=1;samples_seen+=bs
                loss_sum_epoch+=batch_loss*bs;count_epoch+=bs
                if step%a.log_every_steps==0:
                    duration=t4-t0
                    sw.writerow({'experiment':'E0_v2','epoch':epoch_index+1,'step':step,'samples_seen':samples_seen,
                      'elapsed_s':round(elapsed(),4),'loss':round(batch_loss,6),'lr':round(current_lr,9),
                      'batch_samples':bs,'step_s':round(duration,6),'data_s':round(t1-t0,6),
                      'forward_s':round(t2-t1,6),'backward_s':round(t3-t2,6),
                      'optimizer_s':round(t4-t3,6),'sync_s':0,
                      'samples_per_s':round(bs/max(1e-9,duration),2),
                      'gpu_memory_allocated_mb':round(torch.cuda.memory_allocated()/1048576,2) if device.type=='cuda' else 0})
                if step%100==0:print(f'epoch={epoch_index+1}/{a.epochs} step={step} elapsed={elapsed():.1f}s train_loss={batch_loss:.4f} lr={current_lr:.6f}',flush=True)
            if stop_reason=='time_budget':break
            if (epoch_index+1)%a.eval_every_epochs==0 or epoch_index+1==a.epochs:
                acc=evaluate(epoch_index+1)
                checkpoint(epoch_index+1,acc)
            else:
                # Always save latest so resume can restart at a valid epoch boundary.
                checkpoint(epoch_index+1,-1.0)
            if a.target_accuracy>0 and best_acc>=a.target_accuracy:
                print(f'TARGET REACHED: best_validation={best_acc:.4%} at elapsed={target_first:.1f}s; continuing full schedule.',flush=True)
        summary={'experiment':'E0_v2','status':'ok','reason':stop_reason,'steps':step,'samples_seen':samples_seen,
                 'completed_epochs':(epoch_index+1 if stop_reason=='epochs_completed' else epoch_index),
                 'wall_training_s':round(elapsed(),3),'best_validation_accuracy':round(best_acc,6),
                 'target_accuracy':a.target_accuracy,'target_first_s':target_first,'latest_val':last_val,
                 'peak_gpu_memory_mb':round(torch.cuda.max_memory_allocated()/1048576,2) if device.type=='cuda' else 0}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('SUMMARY',json.dumps(summary),flush=True)
    finally:
        sf.close();vf.close()
        if gpuproc:
            gpuproc.terminate()
            try:gpuproc.wait(timeout=5)
            except subprocess.TimeoutExpired:gpuproc.kill();gpuproc.wait()
        if gpuout:gpuout.close()

if __name__=='__main__':main()
