#!/usr/bin/env python3
"""Additional Flower FedAvg per-round figures; original E1_b plot.py remains unchanged."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('result_dir');a=p.parse_args()
    root=Path(a.result_dir);out=root/'plots';out.mkdir(exist_ok=True)
    rows=read(root/'rounds.csv')
    if not rows:raise SystemExit('Missing rounds.csv from Flower server')
    def graphic(filename,title,ykey,ylabel,scale=1.):
        fig,ax=plt.subplots(figsize=(9,4.6))
        x=[int(r['round']) for r in rows]
        y=[float(r[ykey])*scale for r in rows]
        ax.plot(x,y,marker='o',markersize=3)
        ax.set(title=title,xlabel='Federated averaging round',ylabel=ylabel)
        ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/filename,dpi=180);plt.close(fig)
    graphic('40_flower_global_accuracy.png','Flower FedAvg — global-model validation accuracy','global_model_accuracy','Accuracy (%)',100)
    graphic('41_flower_round_communication.png','Flower FedAvg — estimated application payload downloaded','logical_download_bytes','MiB per round',1/(1024**2))
    graphic('42_flower_client_train_time.png','Flower FedAvg — mean local training duration','mean_client_train_seconds','Mean local fit duration (s)')
    graphic('43_flower_aggregator_time.png','Flower FedAvg — server aggregation CPU/host duration','aggregate_host_seconds','Host-side aggregation (s)')
    with (root/'flower_plot_summary.json').open('w') as f:json.dump({'round_count':len(rows),'plots':['40_flower_global_accuracy.png','41_flower_round_communication.png','42_flower_client_train_time.png','43_flower_aggregator_time.png']},f,indent=2)
    print('Flower plots saved to',out)
if __name__=='__main__':main()
