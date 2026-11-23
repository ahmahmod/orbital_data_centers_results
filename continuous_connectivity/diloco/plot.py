#!/usr/bin/env python3
"""E1_b plot generator: all E1_a_new-compatible training/resource figures plus
per-worker gossip/DiLoCo validation and method-specific communications.
Usage: python plot.py results/<run_id> [--gpu-idle-threshold 20]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))


def series(data, x, y, factor=1.):
    points=[]
    for r in data:
        try:points.append((float(r[x]),float(r[y])*factor))
        except (KeyError,TypeError,ValueError):pass
    return ([p[0] for p in points],[p[1] for p in points])


def save_plot(out,name,title,xlabel,ylabel,lines,percent=False):
    fig,ax=plt.subplots(figsize=(9.5,5))
    drew=False
    for title_,xs,ys in lines:
        if xs:
            ax.plot(xs,ys,label=title_,linewidth=1.45,alpha=.90)
            drew=True
    if not drew:
        plt.close(fig)
        return False
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    if percent:ax.set_ylim(0,100)
    ax.grid(alpha=.25)
    if len(lines)>1:ax.legend(loc='best',fontsize=9)
    fig.tight_layout();fig.savefig(out/name,dpi=180);plt.close(fig)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir')
    p.add_argument('--gpu-idle-threshold',type=float,default=20.)
    a=p.parse_args()
    root=Path(a.run_dir);out=root/'plots';out.mkdir(parents=True,exist_ok=True)
    rankfiles=sorted(root.glob('steps_rank*.csv'))
    step={int(f.stem.removeprefix('steps_rank')):rows(f) for f in rankfiles}
    rankval={int(f.stem.removeprefix('validation_rank')):rows(f)
             for f in root.glob('validation_rank*.csv')}
    agg=rows(root/'validation_aggregate.csv')
    legacy_val=rows(root/'validation.csv')
    if not step:raise SystemExit('No steps_rank*.csv files found')

    def rank_lines(y,scale=1.):
        return [(f'Worker {r}',*series(data,'elapsed_s',y,scale)) for r,data in sorted(step.items())]

    # Retained baseline visualizations: directly from original metric fields.
    save_plot(out,'01_train_loss.png','E1_b — local training loss',
        'Training elapsed (s)','Cross-entropy loss',rank_lines('loss'))
    if legacy_val:
        save_plot(out,'02_validation_accuracy.png',
            'E1_b — rank-0 validation accuracy (legacy field)',
            'Training elapsed (s)','Accuracy (%)',
            [('Rank 0',*series(legacy_val,'elapsed_s','val_accuracy',100))],True)
        save_plot(out,'03_validation_loss.png',
            'E1_b — rank-0 validation loss (legacy field)',
            'Training elapsed (s)','Cross-entropy loss',
            [('Rank 0',*series(legacy_val,'elapsed_s','val_loss'))])
    save_plot(out,'04_step_duration.png','E1_b — per-worker step wall time',
        'Training elapsed (s)','Time (ms)',rank_lines('step_s',1000))
    save_plot(out,'05_samples_processed.png','E1_b — samples processed per worker',
        'Training elapsed (s)','Local samples',rank_lines('local_samples'))
    save_plot(out,'06_backward_with_comm.png',
        'E1_b — local backward time (NO DDP per-step AllReduce)',
        'Training elapsed (s)','Time (ms)',rank_lines('backward_with_comm_s',1000))
    save_plot(out,'07_step_throughput.png','E1_b — per-worker step throughput',
        'Training elapsed (s)','Samples/sec',rank_lines('samples_per_s'))
    save_plot(out,'08_gpu_memory_allocated.png','E1_b — allocated GPU tensor memory',
        'Training elapsed (s)','MiB',rank_lines('gpu_memory_allocated_mb'))
    save_plot(out,'09_collective_host_duration.png',
        'E1_b — logged method-specific communication durations',
        'Training elapsed (s)','Host time (ms)',rank_lines('collective_sum_host_s',1000))
    save_plot(out,'10_communication_logical_bytes.png',
        'E1_b — logical communication payload within logged intervals',
        'Training elapsed (s)','MiB',rank_lines('collective_bytes_logical',1/(1024**2)))
    save_plot(out,'11_batch_wait.png','E1_b — batch wait time',
        'Training elapsed (s)','Time (ms)',rank_lines('batch_wait_s',1000))
    save_plot(out,'12_forward.png','E1_b — forward computation',
        'Training elapsed (s)','Time (ms)',rank_lines('forward_s',1000))
    save_plot(out,'13_optimizer.png','E1_b — optimizer step',
        'Training elapsed (s)','Time (ms)',rank_lines('optimizer_s',1000))

    # New: each worker validates its *own* replica on the identical holdout.
    acc_lines=[(f'Worker {r}',*series(data,'elapsed_s','val_accuracy',100))
               for r,data in sorted(rankval.items())]
    if agg:
        acc_lines.append(('Mean of local worker accuracies',*series(agg,'elapsed_s','mean_accuracy',100)))
    save_plot(out,'20_worker_validation.png',
        'E1_b — per-worker validation accuracy (NOT a global model)',
        'Training elapsed (s)','Validation accuracy (%)',acc_lines,True)
    loss_lines=[(f'Worker {r}',*series(data,'elapsed_s','val_loss'))
                for r,data in sorted(rankval.items())]
    if agg:loss_lines.append(('Mean local loss',*series(agg,'elapsed_s','mean_loss')))
    save_plot(out,'21_worker_validation_loss.png',
        'E1_b — per-worker validation loss',
        'Training elapsed (s)','Cross-entropy loss',loss_lines)
    if agg:
        save_plot(out,'22_accuracy_dispersion.png',
            'E1_b — dispersion of local-model validation accuracy',
            'Training elapsed (s)','Accuracy (%)',[
                ('Minimum',*series(agg,'elapsed_s','min_accuracy',100)),
                ('Mean',*series(agg,'elapsed_s','mean_accuracy',100)),
                ('Maximum',*series(agg,'elapsed_s','max_accuracy',100))],True)
        save_plot(out,'23_accuracy_std.png',
            'E1_b — standard deviation across worker accuracies',
            'Training elapsed (s)','Standard deviation (percentage points)',
            [('Cross-worker std',*series(agg,'elapsed_s','std_accuracy',100))])

    # Method-specific message traffic: per-rank and per-direction.
    msgs=[]
    for f in root.glob('messages_rank*.csv'):msgs.extend(rows(f))
    msg_by_rank=defaultdict(list)
    for r in msgs:
        if r.get('event') in ('send_complete','outer_allreduce'):
            try:msg_by_rank[int(r['rank'])].append(r)
            except ValueError:pass
    save_plot(out,'24_completed_communication_payload.png',
        'E1_b — per-event logical payload (NOT network wire bytes)',
        'Training elapsed (s)','Logical payload (MiB)',[
            (f'Rank {k}',*series(v,'elapsed_s','payload_bytes_logical',1/(1024**2)))
            for k,v in sorted(msg_by_rank.items())])

    # Resource sampler from E1_a_new (format may vary across repository revisions).
    for name in sorted(root.glob('resources_rank*.csv')):
        vals=rows(name)
        if not vals:continue
        rank=name.stem.replace('resources_rank','')
        for key,label in [('gpu_util_pct','GPU utilization (%)'),('gpu_utilization_pct','GPU utilization (%)'),
                          ('gpu_memory_used_mb','GPU memory (MB)'),('gpu_power_w','GPU power (W)'),
                          ('net_rx_bytes','Network receive counter (bytes)'),('net_tx_bytes','Network transmit counter (bytes)')]:
            if key in vals[0]:
                xfield='elapsed_s' if 'elapsed_s' in vals[0] else 'unix_time_s'
                save_plot(out,f'30_rank{rank}_{key}.png',f'E1_b — {label}, rank {rank}',
                    xfield,label,[(f'Rank {rank}',*series(vals,xfield,key))])

    report={'experiment':'E1_b','plots':[f.name for f in sorted(out.glob('*.png'))],
            'validation_ranks':sorted(rankval),'n_validation_epochs':len(agg),
            'note':'Aggregate mean is not global-model accuracy. Rootless TCP analysis is separate.'}
    (root/'plot_summary.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {len(report["plots"])} plots to {out}')

if __name__=='__main__':main()
