#!/usr/bin/env python3
"""Plot independently validated gossip worker models. No global model aggregation."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('results_dir');a=p.parse_args()
root=Path(a.results_dir);out=root/'plots';out.mkdir(parents=True,exist_ok=True)
files=sorted(root.glob('validation_rank*.csv'))
if not files:raise SystemExit('No gossip validation_rank*.csv files found')
for field,plot_name,ylabel,mult in [('val_accuracy','20_gossip_worker_accuracy.png','Validation accuracy (%)',100),('val_loss','21_gossip_worker_loss.png','Validation cross-entropy',1)]:
    fig,ax=plt.subplots(figsize=(9,5))
    for file in files:
        with file.open(newline='') as fh:rows=list(csv.DictReader(fh))
        pts=[(float(r['elapsed_s'])/60,float(r[field])*mult) for r in rows if r.get(field)]
        if pts:ax.plot([v[0] for v in pts],[v[1] for v in pts],marker='.',markersize=3,linewidth=1.4,label=file.stem.replace('validation_',''))
    ax.set(xlabel='Elapsed time (minutes)',ylabel=ylabel,title=f'E1_b Gossip — independent {ylabel.lower()} by worker')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/plot_name,dpi=170);plt.close(fig)
print(f'Gossip validation plots saved in {out}')
