#!/usr/bin/env python3
"""E3_1 independent C1/C2 metrics. Never mislabel scheduled risks as observed failures.

Joins scheduler events to E2 validation/step/resource/replay telemetry. Missing
metrics stay null. E2's raw CSVs and existing figures are not overwritten.
"""
import csv,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path

def rows(p):
    if not p.is_file():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))
def num(x):
    try:
        if x is None or x=='':return None
        n=float(x)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None
def write(p,entries,columns):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(entries)
def load(p):
    try:return json.loads(p.read_text())
    except (ValueError,OSError):return {}
def windows(trace,rank,start,duration):
    satellite=f'SAT-DEMO-{rank+1:03d}'
    outages=[]
    for row in rows(trace):
        if row['src']!=satellite or int(row['available'])!=0:continue
        a=num(row['t_start_s']);b=num(row['t_end_s'])
        if a is not None and b is not None and a<duration and b>0:
            outages.append(dict(src=row['src'],dst=row['dst'],begin=start+max(0,a),end=start+min(duration,b)))
    return outages
def union(xs):
    ys=sorted((a,b) for a,b in xs if a is not None and b is not None and b>a)
    merged=[]
    for a,b in ys:
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return merged
def accuracy_series(folder):
    config=load(folder/'e2_run_config.json')
    # Local asynchronous Gossip evaluations are NOT globally simultaneous.
    # Use a fixed, reproducible rank for OFF/ON learning curves; preserve all
    # rank-local files independently for variance/step-aligned analysis.
    if str(config.get('method','')).startswith('gossip'):
        rank0=rows(folder/'validation_rank0.csv')
        series=[(num(r.get('elapsed_s')),num(r.get('val_accuracy'))) for r in rank0]
        series=[x for x in series if None not in x]
        if series:return sorted(series),'async_gossip_rank0_local_validation'
    aggregate=rows(folder/'validation_aggregate.csv')
    if aggregate:
        series=[(num(r.get('elapsed_s')),num(r.get('mean_accuracy'))) for r in aggregate]
        series=[x for x in series if None not in x]
        if series:return sorted(series), 'mean_global_or_worker_aggregate'
    rank0=rows(folder/'validation_rank0.csv')
    series=[(num(r.get('elapsed_s')),num(r.get('val_accuracy'))) for r in rank0]
    series=[x for x in series if None not in x]
    if series:return sorted(series),'rank0_validation'
    legacy=rows(folder/'validation.csv')
    series=[(num(r.get('elapsed_s')),num(r.get('val_accuracy') or r.get('accuracy'))) for r in legacy]
    return sorted(x for x in series if None not in x),'legacy_validation'
def metrics(folder):
    conf=load(folder/'e2_run_config.json')
    if not conf:raise ValueError('Missing E2/E3 run configuration in '+str(folder))
    t0=num(conf.get('init_unix_s'))
    if t0 is None:raise ValueError('Missing shared start time')
    horizon=num(conf.get('duration_s')) or 3600.
    ring_run=conf.get('sync_transport')=='tcp_ring'
    cutoff=t0+horizon
    trace=Path(conf.get('trace_file') or '')
    if not trace.is_file():
        trace=Path(__file__).resolve().parent/'traces/directed_trace_4workers.csv'
    scheduler=[]
    for rank in range(4):scheduler.extend(rows(folder/f'e3_scheduler_decisions_rank{rank}.csv'))
    operations={}
    for e in scheduler:
        cid=e.get('comm_id')
        if not cid:continue
        rec=operations.setdefault(cid,dict(comm_id=cid,rank=e['rank'],method=e.get('method'),
            scope=e.get('scope',''),step=e.get('current_step',''),peer_rank=e.get('peer_rank',''),
            ready_unix=None,start_unix=None,complete_unix=None,apply_unix=None,
            logical_bytes=num(e.get('logical_bytes')) or 0,
            defer_s=0,reported_staleness_s=None,reported_staleness_steps=None,
            status='unknown',rendezvous_wait_s=0,gate_tag='main'))
        action=e.get('scheduler_action','')
        rec['scope']=e.get('scope') or rec['scope']
        if (e.get('note') or '').startswith('gate_tag='):
            rec['gate_tag']=(e.get('note') or '').split('=',1)[1]
        rec['ready_unix']=num(e.get('comm_ready_unix')) or rec['ready_unix']
        rec['start_unix']=num(e.get('comm_start_unix')) or rec['start_unix']
        rec['complete_unix']=num(e.get('comm_complete_unix')) or rec['complete_unix']
        rec['apply_unix']=num(e.get('model_apply_unix')) or rec['apply_unix']
        rec['defer_s']=max(rec['defer_s'],num(e.get('defer_duration_s')) or 0)
        rec['rendezvous_wait_s']=max(rec['rendezvous_wait_s'],num(e.get('rendezvous_wait_s')) or 0)
        rec['logical_bytes']=max(rec['logical_bytes'],num(e.get('logical_bytes')) or 0)
        if action in ('complete','callback_observed'):rec['status']=e.get('communication_status') or 'completed'
        elif action=='failure':rec['status']=e.get('communication_status') or 'failed'
        elif action=='unavailable':rec['status']='not_started_no_window'
        elif action=='defer' and rec['status']=='unknown':rec['status']='deferred_unresolved'
        elif action=='start' and rec['status']=='unknown':rec['status']='started_unresolved'
    if ring_run:
        # Final drain/stop-vote traffic remains in raw CSVs. Censor training
        # operations at the trace horizon, not the later cleanup deadline.
        for cid,op in list(operations.items()):
            if op['gate_tag']=='terminal_vote' or (op['ready_unix'] is not None and op['ready_unix']>=cutoff):
                del operations[cid];continue
            if op['complete_unix'] is not None and op['complete_unix']>cutoff:
                op['complete_unix']=None
                op['status']='censored_at_horizon'
            if op['apply_unix'] is not None and op['apply_unix']>cutoff:
                op['apply_unix']=None
    # E2 direct event logs provide exception / timeout indicators (not IPC statuses).
    legacy=[]
    for rank in range(4):legacy.extend(rows(folder/f'e2_events_rank{rank}.csv'))
    failed=[r for r in legacy if r.get('event') in ('run_failed','comm_error')]
    timeout_events=[r for r in failed if any(x in (r.get('detail','').lower()) for x in ('timeout','timed out','watchdog'))]
    progress_rows={}
    all_progress=[]
    for rank in range(4):
        progress=rows(folder/f'steps_rank{rank}.csv')
        if ring_run:
            progress=[r for r in progress if (num(r.get('unix_time_s') or r.get('unix_s')) or float('inf'))<=cutoff]
        progress_rows[rank]=progress
        for row in progress:
            stamp=num(row.get('unix_time_s') or row.get('unix_s'))
            if stamp is not None:all_progress.append((rank,stamp,num(row.get('step'))))
    completions=[op for op in operations.values() if op['complete_unix'] is not None]
    training_ops=[op for op in operations.values() if not op['gate_tag'].startswith(('preval_','postval_'))]
    training_completions=[op for op in training_ops if op['complete_unix'] is not None]
    outages={rank:windows(trace,rank,t0,horizon) for rank in range(4)}
    # Preserve legacy all-pair outage-onset metric; add true interval overlap
    # on the verified transport-edge union as a separate, nullable metric.
    topology=conf.get('nccl_topology') or {}
    detected=topology.get('detected_edges') or []
    transport_edges={(f'SAT-DEMO-{a+1:03d}',f'SAT-DEMO-{b+1:03d}') for a,b in detected}
    transport_risk=0;transport_attempted=0
    detailed=[];risk=0;attempted=0
    for op in operations.values():
        rank=int(op['rank']);beg=op['start_unix'];end=op['complete_unix']
        src=f'SAT-DEMO-{rank+1:03d}'
        if op['scope']=='collective':
            relevant=[o for rank_events in outages.values() for o in rank_events]
        elif op['scope']=='flower':
            relevant=[o for rank_events in outages.values() for o in rank_events
                      if o['src']=='SAT-DEMO-001' or o['dst']=='SAT-DEMO-001']
        else:
            relevant=[o for o in outages[rank] if
                      op['peer_rank']!='' and o['dst']==f'SAT-DEMO-{int(op["peer_rank"])+1:03d}']
        exposed=bool(beg is not None and any(o['begin']>=beg and (end is None or o['begin']<end) for o in relevant))
        # Exposure indicates *risk*, not confirmed failed traffic.
        if beg is not None:
            attempted+=1
            risk+=int(exposed)
        transport_exposed=None
        if transport_edges and op['scope']=='collective' and beg is not None and end is not None:
            transport_exposed=int(any((o['src'],o['dst']) in transport_edges and o['begin']<end and o['end']>beg for o in relevant))
            transport_attempted+=1
            transport_risk+=transport_exposed
        comm_duration=max(0,end-beg) if end is not None and beg is not None else None
        wall_wait=max(0,end-op['ready_unix']) if end is not None and op['ready_unix'] is not None else None
        detailed.append(dict(**op,comm_duration_s=comm_duration,ready_to_complete_s=wall_wait,
                      exposed_to_scheduled_outage=int(exposed),transport_outage_exposed=transport_exposed))
    fieldnames=list(detailed[0]) if detailed else ['comm_id','rank','method','scope','step','peer_rank','ready_unix','start_unix','complete_unix','apply_unix','logical_bytes','defer_s','reported_staleness_s','reported_staleness_steps','status','rendezvous_wait_s','comm_duration_s','ready_to_complete_s','exposed_to_scheduled_outage','transport_outage_exposed']
    write(folder/'e3_communication_operations.csv',detailed,fieldnames)
    outage_progress=[]
    for rank in range(4):
        own=progress_rows[rank]
        active=union([(o['begin'],o['end']) for o in outages[rank]])
        step_times=[num(r.get('unix_time_s') or r.get('unix_s')) for r in own]
        for begin,end in active:
            n=sum(begin<=t<end for t in step_times if t is not None)
            outage_progress.append(dict(rank=rank,outage_start_unix=begin,outage_end_unix=end,
                        completed_optimizer_steps=n,measurement_scope='union_of_outages_for_worker; no double counting'))
    write(folder/'e3_outage_progress.csv',outage_progress,
        ['rank','outage_start_unix','outage_end_unix','completed_optimizer_steps','measurement_scope'])
    recoveries=[]
    for rank in range(4):
        timestamps=sorted(c['complete_unix'] for c in completions if int(c['rank'])==rank)
        for a,b in union([(o['begin'],o['end']) for o in outages[rank]]):
            follow=next((t for t in timestamps if t>=b),None)
            recoveries.append(dict(rank=rank,outage_end_unix=b,first_success_unix=follow,
                         recovery_time_s=follow-b if follow is not None else None,
                         recovery_observed=follow is not None,
                         scope='rank-level completed operation; not per-link transport recovery'))
    write(folder/'e3_recovery.csv',recoveries,
        ['rank','outage_end_unix','first_success_unix','recovery_time_s','recovery_observed','scope'])
    # C2: Model age from gossip sender snapshot-ready and receiver mixing event.
    sends={}
    for rank in range(4):
        for event in rows(folder/f'e3_scheduler_decisions_rank{rank}.csv'):
            if event.get('scope')=='peer' and event.get('scheduler_action') in ('start','defer'):
                k=(rank,num(event.get('sender_model_version')))
                when=num(event.get('message_created_unix'))
                if when is not None:sends[k]=min(when,sends.get(k,when))
    ages=[]
    for event in legacy:
        if event.get('event')!='model_apply' or not event.get('method','').startswith('gossip'):continue
        detail=dict(x.split('=',1) for x in event.get('detail','').split(';') if '=' in x)
        rec_rank=int(event['rank']);sender=num(detail.get('gossip_from_rank'))
        version=num(detail.get('received_version'))
        applied=num(event.get('unix_s'))
        if sender is None or version is None or applied is None:continue
        created=sends.get((int(sender),version))
        age=applied-created if created is not None else None
        ages.append(dict(rank=rec_rank,sender_rank=int(sender),sender_step=version,
                         receiver_step_at_apply=num(event.get('step')),model_apply_unix=applied,
                         message_created_unix=created,message_age_s=max(0,age) if age is not None else None,
                         message_age_steps=(num(event.get('step'))-version) if num(event.get('step')) is not None else None))
    write(folder/'e3_message_staleness.csv',ages,
        ['rank','sender_rank','sender_step','receiver_step_at_apply','model_apply_unix','message_created_unix','message_age_s','message_age_steps'])
    # A network-wide NIC counter is NOT equivalent to per-message TCP payload.
    network=[];idle=[]
    for rank in range(4):
        records=rows(folder/f'resources_rank{rank}.csv')
        ok=[r for r in records if r.get('status','ok')=='ok']
        tx=[num(r.get('tx_bytes_node')) for r in ok]
        rx=[num(r.get('rx_bytes_node')) for r in ok]
        tx=[x for x in tx if x is not None];rx=[x for x in rx if x is not None]
        network.append(dict(rank=rank,nic_tx_delta_bytes=max(0,tx[-1]-tx[0]) if len(tx)>1 else None,
                            nic_rx_delta_bytes=max(0,rx[-1]-rx[0]) if len(rx)>1 else None,
                            note='node-global NIC counters; not attributable to training process'))
        outwindows=union([(o['begin'],o['end']) for o in outages[rank]])
        outage_samples=[num(r.get('gpu_util_pct')) for r in ok if any(
            a<=float(r.get('unix_time_s','nan'))<b for a,b in outwindows)]
        outage_samples=[v for v in outage_samples if v is not None]
        idle.append(dict(rank=rank,monitored_outage_samples=len(outage_samples),
                         gpu_idle_during_outage_pct=100*sum(v<10 for v in outage_samples)/len(outage_samples) if outage_samples else None,
                         idle_threshold_gpu_util_pct=10,
                         note='sampled GPU utilization proxy; not exclusive communication blocking'))
    write(folder/'e3_network_counters.csv',network,['rank','nic_tx_delta_bytes','nic_rx_delta_bytes','note'])
    write(folder/'e3_gpu_outage_idle.csv',idle,
          ['rank','monitored_outage_samples','gpu_idle_during_outage_pct','idle_threshold_gpu_util_pct','note'])
    # Effective communication interval is the number of optimizer steps between
    # successive completed application-level exchanges of a worker.
    actual_intervals=[]
    for rank in range(4):
        completed_for_rank=sorted(
            [(num(x['step']),x['complete_unix']) for x in completions
              if int(x['rank'])==rank and num(x['step']) is not None
              and not (str(x['method']).startswith('gossip') and x['scope']=='collective')],
            key=lambda x:x[1])
        for prev,next_ in zip(completed_for_rank,completed_for_rank[1:]):
            actual_intervals.append(dict(rank=rank,previous_step=int(prev[0]),
                  completed_step=int(next_[0]),actual_H=int(next_[0]-prev[0]),
                  complete_unix=next_[1]))
    write(folder/'e3_effective_H.csv',actual_intervals,
          ['rank','previous_step','completed_step','actual_H','complete_unix'])
    validation,kind=accuracy_series(folder)
    if ring_run:validation=[(t,a) for t,a in validation if t<=horizon]
    targets={}
    for threshold in (0.5,0.6,0.7):
        found=next((t for t,acc in validation if acc>=threshold),None)
        targets[str(int(threshold*100))]=found
    # For each accuracy milestone, count only successful application payloads
    # completed by that milestone; no unobserved TCP bytes are imputed.
    bytes_to_target={}
    for threshold,elapsed in targets.items():
        if elapsed is None:bytes_to_target[threshold]=None
        else:bytes_to_target[threshold]=sum(op['logical_bytes'] for op in completions
                                                  if op['complete_unix']<=t0+elapsed)
    synced=sorted(x['complete_unix'] for x in completions)
    agevals=sorted(x['message_age_s'] for x in ages if x['message_age_s'] is not None)
    def pct(v,q):
        if not v:return None
        return v[min(len(v)-1,max(0,math.ceil(q*len(v))-1))]
    ring_stats=None
    if ring_run:
        transfers=[row for rank in range(4) for row in rows(folder/f'e3_ring_transfers_rank{rank}.csv')]
        # Only completed chunk exchanges ending before horizon contribute
        # to volume/wait totals. Raw crossing chunks remain available.
        tx=[x for x in transfers if x.get('direction')=='tx']
        in_budget=[x for x in tx if x.get('status')=='completed'
                   and (num(x.get('complete_unix')) or float('inf'))<=cutoff]
        ring_stats=dict(completed_tx_chunks_in_horizon=len(in_budget),
            acknowledged_payload_bytes_in_horizon=sum(num(x.get('payload_bytes')) or 0 for x in in_budget),
            contact_wait_worker_s=sum(num(x.get('contact_wait_s')) or 0 for x in in_budget),
            frames_launched_outside_bidirectional_contact=sum(num(x.get('outside_contact_frames')) or 0 for x in tx),
            acknowledged_after_admitted_window=sum(num(x.get('ack_or_receive_after_window')) or 0 for x in tx),
            post_horizon_tx_frames=sum(num(x.get('post_horizon_frames')) or 0 for x in tx),
            failed_tx_chunks=sum(x.get('status')=='failed' for x in tx),
            measurement_note='Payload excludes TCP headers/retransmissions; volume/wait count whole chunks completed within horizon. Frame launches/overruns are local observations, not proof of on-wire timing.')
    execution=load(folder/'execution_status.json')
    summary=dict(method=conf.get('method'),schedule_mode=conf.get('schedule_mode','off'),
       sync_transport=conf.get('sync_transport'),ring_transfer_metrics=ring_stats,
       measurement_horizon_censored=ring_run,
       run_success=execution.get('success'),failure_reasons=execution.get('failures',[]),
       trace_sha256=conf.get('trace_sha256'),start_unix=t0,duration_s=horizon,
       operation_count=len(detailed),operation_started_count=attempted,
       training_communication_count=len(training_ops),
       training_communication_completed=len(training_completions),
       validation_barrier_operation_count=len(detailed)-len(training_ops),
       operation_completed_count=len(completions),scheduled_outage_exposed_count=risk,
       scheduled_outage_exposure_rate=risk/attempted if attempted else None,
       transport_outage_exposed_count=transport_risk if transport_attempted else None,
       transport_outage_exposure_rate=transport_risk/transport_attempted if transport_attempted else None,
       transport_outage_attempted_count=transport_attempted if transport_edges else None,
       transport_exposure_note='Completed collective interval overlap on detected NCCL ring/ACK edges; null when topology is unverified; risk not failure.',
       observed_failures=len(failed),observed_timeout_events=len(timeout_events),
       total_explicit_deferral_s=sum(x['defer_s'] for x in detailed),
       total_rendezvous_wait_s=sum(x['rendezvous_wait_s'] for x in detailed),
       completed_logical_bytes=sum(x['logical_bytes'] for x in completions),
       logical_bytes_to_target=bytes_to_target,
       actual_H_median=statistics.median(x['actual_H'] for x in actual_intervals) if actual_intervals else None,
       actual_H_range=[min(x['actual_H'] for x in actual_intervals),max(x['actual_H'] for x in actual_intervals)] if actual_intervals else None,
       useful_contact_utilization=None,
       useful_contact_utilization_note='Requires observed attributable on-wire bytes and per-link applied capacity; E2 logical collective sizes are not on-wire volume.',
       successful_transmission_count=len(completions),
       observed_tcp_bytes=None,observed_tcp_bytes_note='No validated per-flow TCP aggregation in this summary; see E2 socket monitoring',
       per_rank_nic_counters=network,
       time_to_accuracy_s=targets,validation_scope=kind,
       final_observed_accuracy=validation[-1][1] if validation else None,
       final_validation_elapsed_s=validation[-1][0] if validation else None,
       total_logged_optimizer_steps=sum(len(x) for x in progress_rows.values()),
       local_steps_during_union_outages=sum(x['completed_optimizer_steps'] for x in outage_progress),
       message_staleness_s_median=statistics.median(agevals) if agevals else None,
       message_staleness_s_p95=pct(agevals,0.95),
       communication_blocked_s_note='E2 ready->complete is NOT pure communication idle; E3 deferral is separately recorded',
       interruption_note='Exposed to scheduled outage is a timestamp overlap risk, NOT a confirmed failed collective.',
       recovery_note='Post-outage first completed rank-level communication; missing observations are censored',
       scheduler_latency_note=('Local TX waits are logged; no per-step central grant or filesystem rendezvous' if ring_run else 'Scheduler rendezvous overhead included explicitly'),
       flower_caveat='Flower server and rank0 client co-located; local traffic bypasses shaped link')
    (folder/'e3_summary.json').write_text(json.dumps(summary,indent=2))
    return summary
def plot(folder):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    out=folder/'e3_plots';out.mkdir(exist_ok=True)
    series,_=accuracy_series(folder)
    conf=load(folder/'e2_run_config.json')
    if conf.get('sync_transport')=='tcp_ring':
        series=[(t,a) for t,a in series if t<=float(conf['duration_s'])]
    if series:
        fig,ax=plt.subplots(figsize=(9,4))
        ax.plot([t/60 for t,a in series],[100*a for t,a in series],marker='o',markersize=3)
        for t in (50,60,70):ax.axhline(t,linestyle=':',linewidth=.8)
        ax.set(xlabel='Elapsed training time (min)',ylabel='Validation accuracy (%)',title='E3_1 learning progress')
        ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'accuracy_vs_time.png',dpi=150);plt.close(fig)
    events=rows(folder/'e3_communication_operations.csv')
    xs=[num(r['ready_unix']) for r in events if num(r.get('ready_unix')) is not None]
    if xs:
        start=min(xs);fig,ax=plt.subplots(figsize=(9,4))
        for r in events:
            st=num(r.get('start_unix'));end=num(r.get('complete_unix'))
            if st is not None and end is not None:ax.plot([(st-start)/60,(end-start)/60],[int(r['rank'])]*2)
        ax.set(xlabel='Time since first communication readiness (min)',ylabel='Worker rank',title='Completed communication intervals')
        ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'communication_timeline.png',dpi=150);plt.close(fig)
def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: python3 e3_analyze.py RESULTS_DIR')
    folder=Path(sys.argv[1]);summary=metrics(folder);plot(folder)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
