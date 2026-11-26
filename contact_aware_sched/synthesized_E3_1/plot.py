#!/usr/bin/env python3
"""Recreate available E1_b-style plots from filter_E2 results, without GPUs.

Usage:
  python plot_e2_like_e1b.py results/E2_gossip8_... 
  python plot_e2_like_e1b.py --all results

Unlike E1_b, E2 does not capture every per-step timing/resource metric.
Unsupported plots are omitted and described in plot_summary.json.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
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


def fnum(row, field):
    try: return float(row[field])
    except (TypeError, ValueError, KeyError): return None


def xy(rows, x, y, scale=1):
    pts = [(fnum(r,x),fnum(r,y)) for r in rows]
    return [(a,b*scale) for a,b in pts if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]


def draw(folder,name,title,ylabel,lines,xlabel='Elapsed time (min)',percent=False):
    lines=[(label,downsample_points(pts)) for label,pts in lines if pts]
    if not lines:return False
    fig,ax=plt.subplots(figsize=(10,5))
    for label,pts in lines:
        ax.plot([x for x,_ in pts],[y for _,y in pts],label=label,linewidth=1.3)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.25)
    if percent:ax.set_ylim(0,100)
    if len(lines)>1:ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(folder/name,dpi=180);plt.close(fig)
    return True


def elapsed_points(rows,field,scale=1):
    return [(t/60,v) for t,v in xy(rows,'elapsed_s',field,scale)]


def derive_interval(rows,metric,per_second=False):
    vals=sorted(xy(rows,'elapsed_s',metric))
    out=[]
    for (t0,v0),(t1,v1) in zip(vals,vals[1:]):
        dt=t1-t0
        if dt>0 and (not per_second or v1>=v0):
            out.append((t1/60,(v1-v0)/dt if per_second else dt*1000))
    return out


def analyze_run(root):
    plots=root/'plots';plots.mkdir(parents=True,exist_ok=True)
    ranks={int(p.stem.split('rank')[-1]):read(p) for p in root.glob('steps_rank*.csv')}
    ranks={k:v for k,v in ranks.items() if v}
    if not ranks:
        print('SKIP (no steps_rank*.csv):',root)
        return
    summary={'experiment':'E2','source':str(root),'plots':[],'missing_telemetry':[],
             'note':'Only measured or explicitly derived E2 metrics are plotted. Completion durations exclude timed-out operations.'}
    def plot(name,title,ylabel,lines,**kwargs):
        if draw(plots,name,title,ylabel,lines,**kwargs):summary['plots'].append(name)
    # E1_b-compatible labels and filenames.
    plot('01_train_loss.png','E2 — local training loss','Cross-entropy loss',
         [(f'Worker {k}',elapsed_points(v,'loss')) for k,v in sorted(ranks.items())])
    plot('04_step_duration.png','E2 — observed interval between logged optimizer steps','Interval (ms)',
         [(f'Worker {k}',derive_interval(v,'step')) for k,v in sorted(ranks.items())])
    plot('05_samples_processed.png','E2 — samples processed by worker','Local samples',
         [(f'Worker {k}',elapsed_points(v,'local_samples')) for k,v in sorted(ranks.items())])
    plot('07_step_throughput.png','E2 — observed training throughput','Samples/sec',
         [(f'Worker {k}',derive_interval(v,'local_samples',True)) for k,v in sorted(ranks.items())])
    # Rank0-only legacy validation, and optionally per-worker validation if added later.
    legacy=read(root/'validation.csv')
    acc_name='accuracy' if any('accuracy' in r for r in legacy) else 'val_accuracy'
    if legacy:
        plot('02_validation_accuracy.png','E2 — rank-0 validation accuracy','Validation accuracy (%)',
             [('Rank 0',elapsed_points(legacy,acc_name,100))],percent=True)
        plot('03_validation_loss.png','E2 — rank-0 validation loss','Cross-entropy loss',
             [('Rank 0',elapsed_points(legacy,'val_loss'))])
    worker_val={int(p.stem.split('rank')[-1]):read(p) for p in root.glob('validation_rank*.csv')}
    if worker_val:
        plot('20_worker_validation.png','E2 — independent local-model validation accuracy',
             'Accuracy (%)',[(f'Worker {k}',elapsed_points(v,'val_accuracy' if v and 'val_accuracy' in v[0] else 'accuracy',100))
                              for k,v in sorted(worker_val.items())],percent=True)
        plot('21_worker_validation_loss.png','E2 — independent local-model validation loss',
             'Cross-entropy loss',[(f'Worker {k}',elapsed_points(v,'val_loss')) for k,v in sorted(worker_val.items())])
    agg=read(root/'validation_aggregate.csv')
    if agg:
        plot('22_accuracy_dispersion.png','E2 — dispersion of worker validation accuracy','Accuracy (%)',
             [(key,elapsed_points(agg,key+'_accuracy',100)) for key in ('min','mean','max')],percent=True)
        plot('23_accuracy_std.png','E2 — std. dev. of worker validation accuracy','Percentage points',
             [('Std',elapsed_points(agg,'std_accuracy',100))])
    # Communication events are timestamped; use run start as reference.
    cfg=json.loads((root/'e2_run_config.json').read_text()) if (root/'e2_run_config.json').exists() else {}
    start=fnum(cfg,'init_unix_s')
    if start is None and ranks:
        r=next(iter(ranks.values()))[0]
        start=fnum(r,'unix_s')-fnum(r,'elapsed_s') if fnum(r,'unix_s') is not None else None
    comm=read(root/'e2_communication_events.csv')
    comm_lines=defaultdict(list)
    if start is not None:
        for row in comm:
            rank=row.get('rank');t=fnum(row,'complete_unix_s');v=fnum(row,'ready_to_complete_s')
            if t is not None and v is not None:comm_lines[rank].append(((t-start)/60,v*1000))
    plot('09_collective_host_duration.png','E2 — completed communication ready-to-complete time','Duration (ms)',
         [(f'Rank {k}',v) for k,v in sorted(comm_lines.items())])
    # Only measured logical bytes, not network wire bytes.
    payload=defaultdict(list)
    if start is not None:
        for path in root.glob('e2_events_rank*.csv'):
            for row in read(path):
                if row.get('event') not in ('comm_ready','comm_complete'):continue
                t=fnum(row,'unix_s');v=fnum(row,'logical_bytes')
                if t is not None and v is not None and row.get('event')=='comm_complete':
                    payload[row['rank']].append(((t-start)/60,v/(1024**2)))
    plot('24_completed_communication_payload.png','E2 — completed logical payload (not wire bytes)','MiB',
         [(f'Rank {k}',v) for k,v in sorted(payload.items())])
    # For E1_b's "10" plot, only event-level bytes are available, not per-step allocations.
    plot('10_communication_logical_bytes.png','E2 — completed communication logical payload','MiB per completed event',
         [(f'Rank {k}',v) for k,v in sorted(payload.items())])
    # Optional E1_b-like resource files, if enabled later.
    resources={int(p.stem.split('rank')[-1]):read(p) for p in root.glob('resources_rank*.csv')}
    for k,vals in sorted(resources.items()):
        if not vals:continue
        xfield='elapsed_s' if 'elapsed_s' in vals[0] else 'unix_time_s'
        for field,label in [('gpu_util_pct','GPU utilization (%)'),('gpu_memory_used_mb','GPU memory (MB)'),('gpu_power_w','GPU power (W)')]:
            if field not in vals[0]:continue
            pts=xy(vals,xfield,field)
            if xfield=='elapsed_s':pts=[(x/60,y) for x,y in pts]
            else:
                if start is None:continue
                pts=[((x-start)/60,y) for x,y in pts]
            plot(f'30_rank{k}_{field}.png',f'E2 — {label}, rank {k}',label,[(f'Rank {k}',pts)])
    summary['missing_telemetry']=[
        '06_backward_with_comm.png (no backward timing in current E2 steps)',
        '08_gpu_memory_allocated.png (no E2 GPU resource sample unless resources_rank*.csv present)',
        '11_batch_wait.png (not logged)',
        '12_forward.png (not logged)',
        '13_optimizer.png (not logged)',
    ]
    (root/'plot_summary.json').write_text(json.dumps(summary,indent=2))
    print(f'{root}: {len(summary["plots"])} E1_b-style plots -> {plots}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path,help='Single run dir, or parent results dir with --all')
    parser.add_argument('--all',action='store_true',help='Plot each E2_* result directory under path')
    args=parser.parse_args()
    if args.all:
        candidates=sorted(p for p in args.path.glob('E2_*') if p.is_dir())
        if not candidates:raise SystemExit('No E2_* run directories found in '+str(args.path))
        for p in candidates:analyze_run(p)
    else:analyze_run(args.path)

if __name__=='__main__':main()
