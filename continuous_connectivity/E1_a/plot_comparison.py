#!/usr/bin/env python3
"""Additional E1_b-aligned DDP figures. Does not replace original E1_a plots."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def records(p):
    if not p.exists():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))
def xy(data,x,y,scale=1):
    pairs=[]
    for r in data:
        try:pairs.append((float(r[x]),float(r[y])*scale))
        except (KeyError,ValueError,TypeError):pass
    return [p[0] for p in pairs],[p[1] for p in pairs]
def plot(path,title,ylabel,lines,percent=False):
    fig,ax=plt.subplots(figsize=(9.5,5));ok=False
    for label,x,y in lines:
        if x:ax.plot(x,y,label=label,linewidth=1.4);ok=True
    if not ok:plt.close(fig);return
    ax.set(xlabel='Training elapsed (s)',ylabel=ylabel,title=title);ax.grid(alpha=.25)
    if percent:ax.set_ylim(0,100)
    if len(lines)>1:ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_dir');a=ap.parse_args();root=Path(a.run_dir)
    dest=root/'plots';dest.mkdir(exist_ok=True)
    ranks={int(p.stem.removeprefix('validation_rank')):records(p) for p in root.glob('validation_rank*.csv')}
    agg=records(root/'validation_aggregate.csv')
    plot(dest/'20_worker_validation.png','E1_a DDP — worker validation accuracy','Validation accuracy (%)',
      [(f'Worker {r}',*xy(rows,'elapsed_s','val_accuracy',100)) for r,rows in sorted(ranks.items())]+
      ([('Mean of worker accuracies',*xy(agg,'elapsed_s','mean_accuracy',100))] if agg else []),True)
    plot(dest/'21_worker_validation_loss.png','E1_a DDP — worker validation loss','Cross-entropy loss',
      [(f'Worker {r}',*xy(rows,'elapsed_s','val_loss')) for r,rows in sorted(ranks.items())]+
      ([('Mean worker loss',*xy(agg,'elapsed_s','mean_loss'))] if agg else []))
    plot(dest/'22_accuracy_dispersion.png','E1_a DDP — cross-replica validation spread','Validation accuracy (%)',
      [(key.title(),*xy(agg,'elapsed_s',key+'_accuracy',100)) for key in ('min','mean','max')],True)
    plot(dest/'23_accuracy_std.png','E1_a DDP — cross-replica validation std','Standard deviation (percentage points)',
      [('Cross-worker std',*xy(agg,'elapsed_s','std_accuracy',100))])
    # Per-event logical volume, not observed bytes transmitted to a peer.
    ev=records(root/'communication_events.csv');byrank={}
    for e in ev:
        byrank.setdefault(e['rank'],[]).append(e)
    # Prefer UNIX timestamps normalized to first event; collective records don't carry elapsed time.
    if ev:
        t0=min(float(e['unix_time_s']) for e in ev)
        lines=[]
        for rank,rows in sorted(byrank.items(),key=lambda x:int(x[0])):
            lines.append((f'Rank {rank}',*[list(t) for t in zip(*[(float(r['unix_time_s'])-t0,float(r['payload_bytes_logical'])/1048576) for r in rows])]))
        plot(dest/'24_completed_communication_payload.png',
          'E1_a DDP — per-bucket logical AllReduce payload (not NIC bytes)','Logical payload (MiB)',lines)
    # Extra plots with E1_b names, retaining E1_a legacy plots.
    steps={int(p.stem.removeprefix('steps_rank')):records(p) for p in root.glob('steps_rank*.csv')}
    plot(dest/'08_gpu_memory_allocated.png','E1_a — GPU memory allocated','MiB',
         [(f'Rank {r}',*xy(rs,'elapsed_s','gpu_memory_allocated_mb')) for r,rs in sorted(steps.items())])
    plot(dest/'10_communication_logical_bytes.png','E1_a — logical DDP bucket bytes in logged steps','MiB',
         [(f'Rank {r}',*xy(rs,'elapsed_s','collective_bytes_logical',1/1048576)) for r,rs in sorted(steps.items())])
    (root/'plot_summary.json').write_text(json.dumps(dict(experiment='E1_a',method='ddp',
        plots=sorted(p.name for p in dest.glob('*.png')),
        validation_ranks=sorted(ranks),n_validation_epochs=len(agg),
        note='DDP cross-replica validation; gradient bucket payload is logical, not network bytes.'),indent=2))
    print('Saved E1_b-aligned E1_a figures and plot_summary.json')
if __name__=='__main__':main()
