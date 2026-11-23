#!/usr/bin/env python3
"""E2 E1_c-style outputs: events, ready-complete intervals, progress in outages.
Communication blocking is an application-level ready->complete proxy, not network-only latency.
"""
import csv,json,pathlib,sys,statistics
from collections import defaultdict

def read(path):
 if not path.exists():return []
 with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
 out=pathlib.Path(sys.argv[1])
 events=sum((read(p) for p in sorted(out.glob('e2_events_rank*.csv'))),[])
 replay=sum((read(p) for p in sorted(out.glob('replay_rank*.csv'))),[])
 starts=defaultdict(list);pairs=[]
 for ev in sorted(events,key=lambda r:float(r['unix_s'])):
  key=(ev['rank'],ev['step'],ev['method'])
  if ev['event']=='comm_ready':starts[key].append(ev)
  if ev['event']=='comm_complete' and starts[key]:
   begin=starts[key].pop(0)
   pairs.append({'rank':ev['rank'],'step':ev['step'],'method':ev['method'],
     'ready_unix_s':begin['unix_s'],'complete_unix_s':ev['unix_s'],
     'ready_to_complete_s':round(float(ev['unix_s'])-float(begin['unix_s']),6)})
 with (out/'e2_communication_events.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['rank','step','method','ready_unix_s','complete_unix_s','ready_to_complete_s']);w.writeheader();w.writerows(pairs)
 outages=[]
 for e in replay:
  if int(e['available'])==0:
   config=json.loads((out/'e2_run_config.json').read_text())
   start=float(config['init_unix_s'])
   outages.append(dict(rank=int(e['rank']),src=e['src'],dst=e['dst'],
     t_start_s=float(e['t_start_s']),t_end_s=float(e['t_end_s']),
     outage_start_unix_s=start+float(e['t_start_s']),outage_end_unix_s=start+float(e['t_end_s']),
     applied_unix_s=e['applied_unix_s']))
 with (out/'e2_actual_outages.csv').open('w',newline='') as f:
  cols=['rank','src','dst','t_start_s','t_end_s','outage_start_unix_s','outage_end_unix_s','applied_unix_s']
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(outages)
 steps=defaultdict(list)
 for ev in events:
  if ev['event']=='local_optimizer_step':steps[int(ev['rank'])].append(float(ev['unix_s']))
 progress=[]
 for e in outages:
  t0=e['outage_start_unix_s'];t1=e['outage_end_unix_s'];r=e['rank']
  progress.append(dict(rank=r,peer=e['dst'],outage_start_unix_s=t0,
    outage_end_unix_s=t1,local_steps_during_outage=sum(t0<=t<t1 for t in steps[r])))
 with (out/'e2_worker_outage_progress.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['rank','peer','outage_start_unix_s','outage_end_unix_s','local_steps_during_outage']);w.writeheader();w.writerows(progress)
 summary={'ranks_seen':sorted(set(e['rank'] for e in events)),'training_step_events':sum(len(v) for v in steps.values()),
  'communications_completed':len(pairs),'mean_comm_ready_to_complete_s':statistics.mean(float(x['ready_to_complete_s']) for x in pairs) if pairs else None,
  'scheduled_outage_intervals_per_direction':len(outages),
  'outage_steps_note':'Per-link progress may count one optimizer step in multiple simultaneous outages.',
  'ready_complete_note':'Ready-to-complete is application-level duration, not pure network blocking.',
  'failure_note':'Missing or incomplete communications can indicate blocked or failed collectives; see training logs.'}
 (out/'e2_analysis.json').write_text(json.dumps(summary,indent=2))
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(12,5));ys=[]
  for r in range(4):
   seq=read(out/f'steps_rank{r}.csv')
   if seq:
    x=[float(row['elapsed_s'])/60 for row in seq];y=[int(row['step']) for row in seq]
    ax.plot(x,y,label=f'W{r+1}')
  ax.set(xlabel='Wall-clock time (minutes)',ylabel='Local optimizer steps',title='E2 training progress');ax.legend();fig.tight_layout();fig.savefig(out/'E2_learning_progress.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(figsize=(10,4))
  for i,(rank,peer) in enumerate(sorted({(e['rank'],e['dst']) for e in outages})):
   for e in outages:
    if (e['rank'],e['dst'])==(rank,peer):ax.broken_barh([(e['t_start_s']/60,(e['t_end_s']-e['t_start_s'])/60)],(i-.4,.8),facecolors='0.35')
  ax.set_yticks(range(len({(e['rank'],e['dst']) for e in outages})),labels=[f"W{k[0]+1}->{k[1][-3:]}" for k in sorted({(e['rank'],e['dst']) for e in outages})]);ax.set(xlabel='Minutes',title='Applied trace outage windows');fig.tight_layout();fig.savefig(out/'E2_outage_timeline.png',dpi=160);plt.close(fig)
 except ImportError:print('Install matplotlib to generate plots')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
