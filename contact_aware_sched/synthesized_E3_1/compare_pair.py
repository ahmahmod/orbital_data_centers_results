#!/usr/bin/env python3
"""Compare one E3_1 schedule-off run against an E3_1 schedule-aware run.
Distinct raw paths and matching method/trace/seed/model are enforced.
Outputs publication-ready descriptive figures, not significance claims.
"""
import argparse,csv,json,math
from pathlib import Path

def read(p):
    if not p.is_file():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))
def obj(p):
    return json.loads(p.read_text())
def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None
def validation(p):
    config=obj(p/'e2_run_config.json')
    def within(v):
        if config.get('sync_transport')=='tcp_ring' or config.get('validation_schedule')=='every_step_checkpoint':
            v=[(t,a) for t,a in v if t<=float(config['duration_s'])]
        return sorted(v)
    if str(config.get('method','')).startswith('gossip'):
        rank=read(p/'validation_rank0.csv')
        v=[(number(r.get('elapsed_s')),number(r.get('val_accuracy'))) for r in rank]
        v=[(t,a) for t,a in v if t is not None and a is not None]
        if v:return within(v)
    agg=read(p/'validation_aggregate.csv')
    if config.get('validation_schedule')=='every_step_checkpoint':
        return within([(float(r['elapsed_s']),float(r['mean_accuracy'])) for r in agg if r.get('complete')=='1' and r.get('within_horizon')=='1'])
    v=[(number(r.get('elapsed_s')),number(r.get('mean_accuracy'))) for r in agg]
    v=[(t,a) for t,a in v if t is not None and a is not None]
    if v:return within(v)
    rank=read(p/'validation_rank0.csv')
    v=[(number(r.get('elapsed_s')),number(r.get('val_accuracy'))) for r in rank]
    v=[(t,a) for t,a in v if t is not None and a is not None]
    return within(v)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--off',type=Path,required=True);ap.add_argument('--aware',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=None)
    a=ap.parse_args()
    if a.off.resolve()==a.aware.resolve():ap.error('different run folders required')
    configs=[obj(p/'e2_run_config.json') for p in (a.off,a.aware)]
    if configs[0].get('schedule_mode')!='off' or configs[1].get('schedule_mode')!='aware':
        ap.error('Expected --off scheduling-off and --aware scheduling-aware results')
    for key in ('method','trace_sha256','dataset','seed','model','duration_s','outer_lr','outer_momentum','training_profile','nccl_algorithm','control_transport','sync_transport','ring_config','validation_schedule','eval_steps'):
        if configs[0].get(key)!=configs[1].get(key):
            ap.error(f'Mismatched {key}: {configs[0].get(key)} vs {configs[1].get(key)}')
    results=[obj(p/'e3_summary.json') for p in (a.off,a.aware)]
    # Default comparisons live next to this technique's results directory.
    # E.g. gossip/results/<run> -> gossip/comparisons/<pair>.
    method=configs[0]['method']
    technique=('ddp' if method=='ddp' else
               'diloco' if method.startswith('diloco') else
               'gossip' if method.startswith('gossip') else 'flower')
    root=Path(__file__).resolve().parent / technique
    default_name='comparison_'+method+'_'+a.aware.name
    out=a.out or (root/'comparisons'/default_name)
    out.mkdir(parents=True,exist_ok=True)
    keys=['run_success','operation_started_count','operation_completed_count','scheduled_outage_exposed_count','scheduled_outage_exposure_rate',
          'observed_failures','observed_timeout_events','total_explicit_deferral_s','total_rendezvous_wait_s',
          'transport_outage_exposed_count','transport_outage_exposure_rate',
          'completed_logical_bytes','successful_transmission_count','local_steps_during_union_outages',
          'total_logged_optimizer_steps','final_observed_accuracy','final_validation_elapsed_s',
          'message_staleness_s_median','message_staleness_s_p95']
    with (out/'comparison_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','off','aware','aware_minus_off','relative_change_pct'])
        for k in keys:
            x=results[0].get(k);y=results[1].get(k)
            xn=number(x);yn=number(y)
            delta=(yn-xn) if xn is not None and yn is not None else None
            pct=(100*delta/xn) if delta is not None and xn else None
            w.writerow([k,x,y,delta,pct])
        for k in ('ack_payload_in_horizon','partial_contained_bytes','partial_ack_bytes','full_contained_bytes','crossing_bytes','partial_contained_fraction','max_outstanding_frames'):
            x=number((results[0].get('ring_transfer_metrics') or {}).get(k));y=number((results[1].get('ring_transfer_metrics') or {}).get(k))
            w.writerow([k,x,y,y-x if None not in (x,y) else None,100*(y-x)/x if None not in (x,y) and x else None])
        for threshold in ('50','60','70'):
            x=number(results[0].get('time_to_accuracy_s',{}).get(threshold))
            y=number(results[1].get('time_to_accuracy_s',{}).get(threshold))
            w.writerow(['time_to_accuracy_'+threshold,x,y,(y-x) if x is not None and y is not None else None,
                        100*(y-x)/x if x is not None and y is not None and x>0 else None])
    report=dict(method=configs[0]['method'],comparison_mode=configs[1].get('schedule_mode'),compared_runs=[str(a.off),str(a.aware)],
       matching_trace_sha=configs[0].get('trace_sha256'),
       warning='Descriptive comparison only. Check training status, measurement scope and run repetition.',
       off=results[0],aware=results[1])
    (out/'comparison_summary.json').write_text(json.dumps(report,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Wrote comparison CSV/JSON; matplotlib absent')
        return
    a0=validation(a.off);a1=validation(a.aware)
    fig,ax=plt.subplots(figsize=(9,5))
    for title,series in [('Reactive / OFF',a0),('Contact-aware / '+configs[1].get('schedule_mode','aware').upper(),a1)]:
        if series:ax.plot([t/60 for t,_ in series],[acc*100 for _,acc in series],marker='o',markersize=2,label=title)
    for t in (50,60,70):ax.axhline(t,linewidth=.7,linestyle=':',alpha=.5)
    ax.set(xlabel='Elapsed wall-clock (minutes)',ylabel='Validation accuracy (%)',title='E3_1: accuracy versus time')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'accuracy_vs_time.png',dpi=170);plt.close(fig)
    categories=['Started','Completed','Outage exposed','Failure events','Timeout events']
    fields=['operation_started_count','operation_completed_count','scheduled_outage_exposed_count','observed_failures','observed_timeout_events']
    fig,ax=plt.subplots(figsize=(10,4))
    xs=list(range(len(categories)))
    ax.bar([x-.2 for x in xs],[results[0].get(k) or 0 for k in fields],.4,label='OFF')
    ax.bar([x+.2 for x in xs],[results[1].get(k) or 0 for k in fields],.4,label='ON')
    ax.set_xticks(xs,categories,rotation=15)
    ax.set(ylabel='Count',title='Operation counts, risks, and observed failures')
    ax.legend();fig.tight_layout();fig.savefig(out/'communication_outcomes.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for k,title in [('total_explicit_deferral_s','Deferral'),('total_rendezvous_wait_s','Coordination')]:
        ax.bar([0,1],[results[i].get(k) or 0 for i in range(2)],alpha=.5,label=title)
    ax.set_xticks([0,1],['OFF','ON'])
    ax.set(ylabel='Accumulated worker-seconds (overlap possible)',title='Scheduler-induced waits')
    ax.legend();fig.tight_layout();fig.savefig(out/'scheduler_waits.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for folder,title in [(a.off,'OFF'),(a.aware,'ON')]:
        ages=[number(r.get('message_age_s')) for r in read(folder/'e3_message_staleness.csv')]
        ages=sorted(x for x in ages if x is not None)
        if ages:
            ax.plot([(i+.5)/len(ages) for i in range(len(ages))],ages,label=title)
    ax.set(xlabel='Empirical quantile',ylabel='Delivered gossip message age (s)',
        title='Message staleness: delivered updates only')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'gossip_staleness.png',dpi=170);plt.close(fig)
    # C2: same learning quality against successful logical communication payload.
    fig,ax=plt.subplots(figsize=(9,5))
    for folder,title,config in [(a.off,'OFF',configs[0]),(a.aware,'ON',configs[1])]:
        series=validation(folder)
        completed=sorted(
          (number(row.get('complete_unix')),number(row.get('logical_bytes')) or 0.0)
          for row in read(folder/'e3_communication_operations.csv')
          if number(row.get('complete_unix')) is not None)
        start=number(config.get('init_unix_s')) or 0
        volume=[];accuracies=[]
        for elapsed,accuracy in series:
            total=sum(value or 0 for stamp,value in completed if stamp<=start+elapsed)
            volume.append(total/1024**3);accuracies.append(accuracy*100)
        if volume:ax.plot(volume,accuracies,marker='o',markersize=2,label=title)
    ax.set(xlabel='Cumulative completed logical payload (GiB)',
           ylabel='Validation accuracy (%)',title='Accuracy vs successful logical communication')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'accuracy_vs_logical_bytes.png',dpi=170);plt.close(fig)

    # C2 instrumentation sanity: effective H must remain fixed in E3_1,
    # except for skipped gossip sends or final/boundary effects.
    fig,ax=plt.subplots(figsize=(9,4))
    for folder,title in [(a.off,'OFF'),(a.aware,'ON')]:
        records=read(folder/'e3_effective_H.csv')
        xy=[(number(x.get('completed_step')),number(x.get('actual_H'))) for x in records]
        xy=[(x,y) for x,y in xy if x is not None and y is not None]
        if xy:ax.scatter([x for x,y in xy],[y for x,y in xy],s=8,alpha=.45,label=title)
    ax.set(xlabel='Local optimizer step',ylabel='Observed steps between completed exchanges',
        title='Effective communication interval H (no adaptive H in E3_1)')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'effective_H_over_time.png',dpi=170);plt.close(fig)

    # C1: censored recovery times are excluded from this distribution but
    # remain explicitly counted in e3_recovery.csv.
    fig,ax=plt.subplots(figsize=(9,4))
    for folder,title in [(a.off,'OFF'),(a.aware,'ON')]:
        v=sorted(number(x.get('recovery_time_s')) for x in read(folder/'e3_recovery.csv')
                 if number(x.get('recovery_time_s')) is not None)
        if v:ax.plot(v,[(i+1)/len(v) for i in range(len(v))],label=title)
    ax.set(xlabel='First successful rank-level communication after outage (s)',
           ylabel='Empirical CDF (observed recoveries only)',
           title='Recovery-time distribution (unobserved recoveries are censored)')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'recovery_time_cdf.png',dpi=170);plt.close(fig)
    print('Wrote',out)
if __name__=='__main__':main()
