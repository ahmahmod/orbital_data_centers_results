#!/usr/bin/env python3
"""Reproduce available E1_b plot families from E2 telemetry; aggregate worker validation.
Usage: python3 plot_e2_full.py results/E2_method_...; or python3 plot_e2_full.py results --all
Requires matplotlib. No GPUs/OAR reservation needed.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def _positive_env(name, default):
    """Positive integer tuning knob for large experimental traces."""
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_PLOT_POINTS = _positive_env('E2_PLOT_MAX_POINTS', 3000)
STEP_STRIDE = _positive_env('E2_PLOT_STEP_STRIDE', 200)
COMM_STRIDE = _positive_env('E2_PLOT_COMM_STRIDE', 20)
EVENT_STRIDE = _positive_env('E2_PLOT_EVENT_STRIDE', 20)


def downsample_points(points):
    """Cap memory handed to Matplotlib while preserving the final observation."""
    if len(points) <= MAX_PLOT_POINTS:
        return points
    stride = max(1, math.ceil((len(points) - 1) / (MAX_PLOT_POINTS - 1))) if MAX_PLOT_POINTS > 1 else len(points)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        if len(sampled) >= MAX_PLOT_POINTS:
            sampled[-1] = points[-1]
        else:
            sampled.append(points[-1])
    return sampled


def read(path):
    """Stream large per-step/event CSVs instead of materializing every row."""
    if not path.is_file():
        return []

    name = path.name
    rows = []
    last_selected = None
    eligible = 0
    if name.startswith('steps_rank'):
        stride = STEP_STRIDE
    elif name == 'e2_communication_events.csv':
        stride = COMM_STRIDE
    elif name.startswith('e2_events_rank'):
        stride = EVENT_STRIDE
    else:
        stride = 1

    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            # This plotter uses only completed exchanges from the large event logs.
            if name.startswith('e2_events_rank') and row.get('event') != 'comm_complete':
                continue
            last_selected = row
            if eligible % stride == 0:
                rows.append(row)
            eligible += 1

    if last_selected is not None and (not rows or rows[-1] != last_selected):
        rows.append(last_selected)
    return rows


def number(r, key):
    try:
        v=float(r[key]);return v if math.isfinite(v) else None
    except (KeyError,TypeError,ValueError):return None


def xy(rows, xkey, ykey, mult=1):
    pts=[(number(r,xkey),number(r,ykey)) for r in rows]
    return [(x,y*mult) for x,y in pts if x is not None and y is not None]


def save(out, name, title, ylabel, groups, percent=False):
    fig, ax=plt.subplots(figsize=(10,5.2))
    drawn=0
    for label, pts in groups:
        pts = downsample_points(pts)
        if pts:
            ax.plot([p[0] for p in pts],[p[1] for p in pts], label=label, linewidth=1.2, alpha=.85)
            drawn+=1
    if not drawn:
        plt.close(fig);return False
    ax.set(title=title,xlabel='Elapsed time (seconds)',ylabel=ylabel)
    if percent: ax.set_ylim(bottom=0,top=100)
    ax.grid(alpha=.22)
    if drawn>1:ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(out/name,dpi=175);plt.close(fig)
    return True


def aggregate_validation(root,interval):
    """Align independent checks by scheduled time slot; never require a barrier."""
    by_slot=defaultdict(dict)
    config_path=root/'e2_run_config.json'
    config=json.loads(config_path.read_text()) if config_path.is_file() else {}
    by_epoch=(config.get('validation_schedule')=='every_epoch')
    for f in sorted(root.glob('validation_rank*.csv')):
        try:rank=int(f.stem.removeprefix('validation_rank'))
        except ValueError:continue
        for v in read(f):
            elapsed=number(v,'elapsed_s');acc=number(v,'val_accuracy');loss=number(v,'val_loss')
            if elapsed is None or acc is None or loss is None: continue
            # E1_b compares the same completed epoch across ranks; the older
            # E2 wall-clock mode instead aligns independent checks by time.
            slot=int(v['epoch']) if by_epoch else max(1, round(elapsed/interval))
            by_slot[slot][rank]=(v,acc,loss)
    fields=['experiment','method','epoch','step','elapsed_s','unix_time_s','mean_accuracy',
            'min_accuracy','max_accuracy','std_accuracy','mean_loss','min_loss','max_loss',
            'std_loss','val_samples_per_worker','world_size','mean_accuracy_target_reached',
            'all_workers_target_reached','observed_ranks','complete']
    rows=[]
    for slot,ranks in sorted(by_slot.items()):
        aa=[x[1] for x in ranks.values()];ll=[x[2] for x in ranks.values()]
        first=next(iter(ranks.values()))[0]
        record=dict(experiment='E2',method=first.get('method',''),epoch=slot,
           step=first.get('step',''),
           elapsed_s=(statistics.mean(float(x[0]['elapsed_s']) for x in ranks.values()) if by_epoch else slot*interval),
           unix_time_s=first.get('unix_time_s',''),
           mean_accuracy=statistics.mean(aa),min_accuracy=min(aa),max_accuracy=max(aa),
           std_accuracy=statistics.pstdev(aa),mean_loss=statistics.mean(ll),
           min_loss=min(ll),max_loss=max(ll),std_loss=statistics.pstdev(ll),
           val_samples_per_worker=first.get('val_samples',''),world_size=4,
           mean_accuracy_target_reached='',all_workers_target_reached='',
           observed_ranks=','.join(map(str,sorted(ranks))),complete=int(len(ranks)==4))
        rows.append(record)
    with (root/'validation_aggregate.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
    return rows


def plot_run(root):
    root=pathlib.Path(root)
    steps={i:read(root/f'steps_rank{i}.csv') for i in range(4)}
    if not any(steps.values()):
        print('SKIP (no training steps):',root);return
    config=json.loads((root/'e2_run_config.json').read_text()) if (root/'e2_run_config.json').is_file() else {}
    interval=float(config.get('eval_interval_s',300) or 300)
    agg=aggregate_validation(root,interval)
    val={i:read(root/f'validation_rank{i}.csv') for i in range(4)}
    resources={i:read(root/f'resources_rank{i}.csv') for i in range(4)}
    comm=read(root/'e2_communication_events.csv')
    events={i:read(root/f'e2_events_rank{i}.csv') for i in range(4)}
    out=root/'plots';out.mkdir(parents=True,exist_ok=True)
    created=[]
    def plot(name,title,ylabel,groups,percent=False):
        if save(out,name,title,ylabel,groups,percent):created.append(name)
    def step_groups(field,mult=1):
        return [(f'Worker {r}',xy(seq,'elapsed_s',field,mult)) for r,seq in steps.items()]
    def val_groups(field,mult=1):
        return [(f'Worker {r}',xy(seq,'elapsed_s',field,mult)) for r,seq in val.items()]
    def res_groups(field):
        return [(f'Worker {r}',xy(seq,'elapsed_s',field)) for r,seq in resources.items()]
    plot('01_train_loss.png','E2 — local training loss','Cross-entropy',step_groups('loss'))
    plot('02_validation_accuracy.png','E2 — rank-0 independent validation accuracy','Accuracy (%)',
         [('Rank 0',xy(val[0],'elapsed_s','val_accuracy',100))],True)
    plot('03_validation_loss.png','E2 — rank-0 independent validation loss','Cross-entropy',
         [('Rank 0',xy(val[0],'elapsed_s','val_loss'))])
    for name,title,field,mult,label in [
     ('04_step_duration.png','Step wall duration','step_s',1000,'Time (ms)'),
     ('05_samples_processed.png','Local samples processed','local_samples',1,'Samples'),
     ('06_backward_with_comm.png','Backward host duration (DDP all-reduce logged separately)','backward_with_comm_s',1000,'Time (ms)'),
     ('07_step_throughput.png','Local training throughput','samples_per_s',1,'Samples/s'),
     ('08_gpu_memory_allocated.png','Allocated GPU tensor memory','gpu_memory_allocated_mb',1,'MiB'),
     ('10_communication_logical_bytes.png','Synchronous communication bytes per step (Gossip separate)','collective_bytes_logical',1/1048576,'Logical MiB'),
     ('11_batch_wait.png','Training data batch wait','batch_wait_s',1000,'Time (ms)'),
     ('12_forward.png','Forward pass host duration','forward_s',1000,'Time (ms)'),
     ('13_optimizer.png','Optimizer host duration','optimizer_s',1000,'Time (ms)')]:
        plot(name,'E2 — '+title,label,step_groups(field,mult))
    if comm:
        initial=float(config.get('init_unix_s',0))
        by_rank=defaultdict(list)
        for row in comm:
            x=number(row,'complete_unix_s');y=number(row,'ready_to_complete_s')
            if x is not None and y is not None: by_rank[row['rank']].append((x-initial,y*1000))
        plot('09_collective_host_duration.png','E2 — application-level communication ready-to-complete','Time (ms)',
             [(f'Worker {i}',by_rank.get(str(i),[])) for i in range(4)])
    plot('20_worker_validation.png','E2 — independent local-model validation accuracy','Accuracy (%)',
         val_groups('val_accuracy',100)+[('Mean available ranks',xy(agg,'elapsed_s','mean_accuracy',100))],True)
    plot('21_worker_validation_loss.png','E2 — independent local-model validation loss','Cross-entropy',
         val_groups('val_loss')+[('Mean available ranks',xy(agg,'elapsed_s','mean_loss'))])
    plot('22_accuracy_dispersion.png','E2 — independent worker accuracy dispersion','Accuracy (%)',
         [(name,xy(agg,'elapsed_s',col,100)) for name,col in [('Min','min_accuracy'),('Mean','mean_accuracy'),('Max','max_accuracy')]],True)
    plot('23_accuracy_std.png','E2 — standard deviation of worker accuracy','Percentage points',
         [('Worker standard deviation',xy(agg,'elapsed_s','std_accuracy',100))])
    by_rank=defaultdict(list)
    for i,records in events.items():
        for row in records:
            if row.get('event')=='comm_complete':
                t=number(row,'unix_s');v=number(row,'logical_bytes')
                if t is not None and v is not None:by_rank[i].append((t-float(config.get('init_unix_s',0)),v/1048576))
    plot('24_completed_communication_payload.png','E2 — completed logical communication payload','MiB',
         [(f'Worker {i}',by_rank[i]) for i in range(4)])
    for i,rec in resources.items():
        for field,ylabel,unit in [('gpu_util_pct','GPU utilization (%)','%'),
               ('gpu_mem_used_mib','GPU memory used (MiB)','MiB'),('gpu_power_w','GPU power (W)','W'),
               ('rx_bytes_node','NIC cumulative RX bytes','bytes'),('tx_bytes_node','NIC cumulative TX bytes','bytes')]:
            plot(f'30_rank{i}_{field}.png',f'E2 — rank {i} {ylabel}',ylabel,
                 [(f'Rank {i}',xy(rec,'elapsed_s',field))],unit=='%')
    # Collective events on E2 represent logical tensor payloads, not observed wire bytes.
    report=dict(experiment='E2',method=config.get('method'),plots=created,
      validation_ranks=[i for i in range(4) if val[i]],validation_slots=len(agg),
      complete_validation_slots=sum(int(r['complete']) for r in agg),
      note='Validation is independently computed for each worker. Means are NOT global model accuracy. '
           'Gossip async communication is counted from event logs, not attributed to optimizer steps. '
           'GPU/host timestamps are samples; timing instrumentation adds synchronization overhead.')
    (root/'plot_summary.json').write_text(json.dumps(report,indent=2))
    print(root,'generated',len(created),'plots; validation slots:',len(agg),
          'complete 4-worker slots:',report['complete_validation_slots'])


def main():
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--all',action='store_true')
    a=p.parse_args();path=pathlib.Path(a.path)
    if a.all:
        for d in sorted(path.glob('E2_*')):
            if d.is_dir() and (d/'e2_run_config.json').exists():plot_run(d)
    else:plot_run(path)
if __name__=='__main__':main()
