#!/usr/bin/env python3
"""E1_a: evidence-based plots + per-rank summary from instrumented CSV files."""
import argparse,csv,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def records(path):
    if not path.exists():return []
    with path.open(newline='') as f:return list(csv.DictReader(f))

def point(row,key):
    try:return float(row[key])
    except (KeyError,ValueError,TypeError):return None

def xy(rows,x,y):
    data=[(point(r,x),point(r,y)) for r in rows]
    valid=[(a,b) for a,b in data if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    return [a for a,b in valid],[b for a,b in valid]

def render(path,series,xlabel,ylabel,title,ylim=None,events=None):
    fig,ax=plt.subplots(figsize=(11,4.6))
    for label,x,y in series:
        if x:ax.plot(x,y,label=label,linewidth=1.05,alpha=.85)
    if events:
        for i,(start,end) in enumerate(events):ax.axvspan(start,end,color='grey',alpha=.14,label='scheduled outage' if i==0 else None)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    if ylim:ax.set_ylim(*ylim)
    ax.grid(alpha=.22)
    if len(series)>1 or events:ax.legend(loc='best',fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def read_summary(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir');p.add_argument('--gpu-idle-threshold',type=float,default=20.)
    a=p.parse_args();root=Path(a.run_dir);out=root/'plots';out.mkdir(parents=True,exist_ok=True)
    ranks=sorted(int(f.stem.replace('steps_rank','')) for f in root.glob('steps_rank*.csv') if f.stem.replace('steps_rank','').isdigit())
    streams=[records(root/f'steps_rank{rank}.csv') for rank in ranks]
    resources=[records(root/f'resources_rank{rank}.csv') for rank in ranks]
    val=records(root/'validation.csv')
    if not any(streams):raise SystemExit('No steps_rank*.csv data found; E1_a run must produce training logs first')

    def lines(y,label=None,mult=1.):
        result=[]
        for i,rows in enumerate(streams):
            x,v=xy(rows,'elapsed_s',y)
            result.append((f'rank {ranks[i]}',x,[n*mult for n in v]))
        return result
    render(out/'01_training_loss.png',lines('loss'),'Elapsed training time (s)','Cross entropy loss','E1_a — local training loss')
    if val:
        x,v=xy(val,'elapsed_s','val_accuracy')
        render(out/'02_validation_accuracy.png',[('validation',x,[100*z for z in v])], 'Elapsed training time (s)','Validation accuracy (%)','E1_a — validation accuracy')
        render(out/'03_validation_loss.png',[('validation',*xy(val,'elapsed_s','val_loss'))], 'Elapsed training time (s)','Cross entropy loss','E1_a — validation loss')
    for fname,key,units,title,mult in [
       ('04_step_duration.png','step_s','ms','Step execution duration (batch available to optimizer)',1000),
       ('05_batch_wait.png','batch_wait_s','ms','Host-side wait for next training minibatch',1000),
       ('06_forward.png','forward_s','ms','Forward pass duration',1000),
       ('07_backward_with_comm.png','backward_with_comm_s','ms','Backward includes overlapping gradient all-reduce',1000),
       ('08_optimizer.png','optimizer_s','ms','Optimizer update duration',1000),
       ('09_collective_span.png','collective_span_host_s','ms','NCCL future host span (not isolated NCCL GPU time)',1000),
       ('10_collective_sum.png','collective_sum_host_s','ms','Sum of NCCL future durations (buckets may overlap)',1000),
       ('11_collective_bytes.png','collective_bytes_logical','MiB/step','Per-rank logical collective payload (not NIC wire bytes)',1/1048576),
       ('12_step_throughput.png','samples_per_s','images/s/rank','Per-rank throughput while step executes',1),
       ('13_samples_processed.png','local_samples','local sample presentations','Sample progress per rank',1)]:
        render(out/fname,lines(key,mult=mult),'Elapsed training time (s)',units,f'E1_a — {title}')
    render(out/'14_gpu_utilization.png',[(f'rank {ranks[i]}',*xy(rows,'elapsed_s','gpu_util_pct')) for i,rows in enumerate(resources)],'Elapsed since rank start (s)','GPU utilization (%)','E1_a — GPU utilization by rank',ylim=(0,105))
    render(out/'15_gpu_power.png',[(f'rank {ranks[i]}',*xy(rows,'elapsed_s','gpu_power_w')) for i,rows in enumerate(resources)],'Elapsed since rank start (s)','GPU power (W)','E1_a — GPU power samples')
    # Node interface byte counters are cumulative and include control/background traffic.
    network=[]
    for i,rows in enumerate(resources):
        if not any(r.get('rx_bytes_node') for r in rows):continue
        for axis in ('tx_bytes_node','rx_bytes_node'):
            x,v=xy(rows,'elapsed_s',axis)
            if x:
                network.append((f'rank {i} host {axis.split("_")[0]}',x,[(b-v[0])/1048576 for b in v]))
    if network:render(out/'16_node_nic_counters.png',network,'Elapsed since rank start (s)','MiB since first observation', 'E1_a — Node-global interface traffic (includes non-training traffic)')

    report=[]
    for rank,(steps,rr) in enumerate(zip(streams,resources)):
        timing=[point(r,'step_s') for r in steps if point(r,'step_s') is not None]
        comm=[point(r,'collective_span_host_s') for r in steps if point(r,'collective_span_host_s') is not None]
        util=[point(r,'gpu_util_pct') for r in rr if point(r,'gpu_util_pct') is not None]
        logical=[point(r,'collective_bytes_logical') for r in steps if point(r,'collective_bytes_logical') is not None]
        report.append(dict(rank=ranks[rank],logged_steps=len(steps),final_logged_step=int(float(steps[-1]['step'])) if steps else None,
            median_step_ms=round(statistics.median(timing)*1000,3) if timing else None,
            p95_step_ms=round(sorted(timing)[min(len(timing)-1,int(.95*len(timing)))]*1000,3) if timing else None,
            median_collective_host_span_ms=round(statistics.median(comm)*1000,3) if comm else None,
            median_logical_bytes_per_logged_step=round(statistics.median(logical)) if logical else None,
            gpu_samples=len(util),gpu_mean_utilization_pct=round(statistics.mean(util),3) if util else None,
            gpu_idle_sample_pct=round(100*sum(v<a.gpu_idle_threshold for v in util)/len(util),3) if util else None))
    info={'experiment':'E1_a','gpu_idle_threshold_pct':a.gpu_idle_threshold,
          'gpu_idle_note':'fraction of 1s samples below utilization threshold, NOT actual GPU idle time',
          'network_note':'NIC counters are node-global; cannot distinguish DDP and management traffic',
          'communication_note':'Collective future timings measure host scheduling/completion, not exclusive GPU collective kernel times',
          'rank_statistics':report,'training_summary':read_summary(root/'summary.json')}
    (root/'analysis.json').write_text(json.dumps(info,indent=2))
    print(f'Generated {len(list(out.glob("*.png")))} plot files in {out} and analysis.json')

if __name__=='__main__':main()
