"""Shared-filesystem startup barrier for E3 DDP; never shape before DDP is ready.

A distinct result directory identifies each run, and run_id prevents stale
files from accidentally releasing a rank. All writes are atomic renames.
"""
import json
import os
from pathlib import Path
import time

def startup_dir(output):
    return Path(output)/'.e3_control'/'startup'

def atomic_json(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid()))
    try:
        tmp.write_text(json.dumps(payload))
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

def publish_ready(output,rank,run_id):
    atomic_json(startup_dir(output)/f'ready_rank{int(rank)}.json',
                {'rank':int(rank),'run_id':str(run_id),'ready_unix':time.time()})

def ready_status(output,run_id,world=4):
    found={}
    for rank in range(world):
        path=startup_dir(output)/f'ready_rank{rank}.json'
        try:
            item=json.loads(path.read_text())
            if item.get('run_id')==run_id and item.get('rank')==rank:
                found[rank]=item
        except (OSError,ValueError,TypeError):
            continue
    return found

def publish_start(output,run_id,start_unix):
    if len(ready_status(output,run_id))!=4:
        raise RuntimeError('Refusing to start contact trace before all four DDP workers are ready')
    atomic_json(startup_dir(output)/'start.json',
                {'run_id':str(run_id),'start_unix':float(start_unix)})

def await_start(output,run_id,timeout_s=1800,poll_s=.1):
    deadline=time.monotonic()+float(timeout_s)
    path=startup_dir(output)/'start.json'
    while time.monotonic()<deadline:
        try:
            obj=json.loads(path.read_text())
            if obj.get('run_id')==run_id:
                start=float(obj['start_unix'])
                if start<=0:raise ValueError('invalid scheduled start')
                return start
        except (OSError,ValueError,KeyError,TypeError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f'Coordinator did not publish DDP start within {timeout_s}s; '
                       f'run_id={run_id}, startup_dir={path.parent}')

def publish_failure(output,rank,run_id,detail):
    atomic_json(startup_dir(output)/f'failed_rank{int(rank)}.json',
                {'rank':int(rank),'run_id':str(run_id),'failed_unix':time.time(),
                 'detail':str(detail)[-1500:]})

def failure_status(output,run_id,world=4):
    found={}
    for rank in range(world):
        path=startup_dir(output)/f'failed_rank{rank}.json'
        try:
            item=json.loads(path.read_text())
            if item.get('run_id')==run_id and item.get('rank')==rank:
                found[rank]=item
        except (OSError,ValueError,TypeError):
            continue
    return found
