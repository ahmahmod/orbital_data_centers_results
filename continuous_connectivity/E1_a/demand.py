#!/usr/bin/env python3
"""Summarize DDP logical communication demand, worker mapping, topology, and timing.
Never assert logical bucket bytes = physical per-peer network bytes.
"""
import argparse,csv,json,math,re,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
 if not path.is_file():return []
 with path.open(newline='') as f:return list(csv.DictReader(f))
def number(row,k,default=None):
 try:return float(row[k])
 except (KeyError,ValueError,TypeError):return default
def percentile(v,p):
 if not v:return None
 v=sorted(v);a=(len(v)-1)*p/100;i=int(a);q=a-i
 return v[i]*(1-q)+v[min(i+1,len(v)-1)]*q
def describe(v):
 return {'count':len(v),'min':min(v) if v else None,'median':percentile(v,50),'p95':percentile(v,95),'max':max(v) if v else None}
def graph(filename,x,y,xlabel,ylabel,title):
 if not x:return
 plt.figure(figsize=(10,4.25));plt.plot(x,y,linewidth=1.25)
 plt.xlabel(xlabel);plt.ylabel(ylabel);plt.title(title);plt.grid(alpha=.25);plt.tight_layout();plt.savefig(filename,dpi=170);plt.close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_dir');a=ap.parse_args()
 root=Path(a.run_dir);out=root/'plots';out.mkdir(parents=True,exist_ok=True)
 mapping=json.loads((root/'worker_map.json').read_text())['workers'] if (root/'worker_map.json').exists() else []
 identity={int(p['rank']):p for p in mapping}
 ranks=sorted(identity) if identity else sorted(int(f.stem.replace('steps_rank','')) for f in root.glob('steps_rank*.csv'))
 records=[];steps=[];barriers=[]
 for rank in ranks:
  records+=read(root/f'collectives_rank{rank}.csv')
  steps+=read(root/f'steps_rank{rank}.csv')
  barriers+=read(root/f'sync_events_rank{rank}.csv')
 # Explicitly report logical bucket events only. Ranks are collective members, not destinations.
 result={'experiment':'E1_a','scope':'logical_allreduce_demand',
         'definitions':{'bytes':'one input gradient bucket per participating rank, NOT physical bytes on wire',
         'transfer':'one rank submitting one DDP bucket collective, NOT a unicast message',
         'frequency':'observed collective bucket submissions per second in each rank',
         'blocked_time':'epoch barriers are directly measured; host-side collective futures are overlapped and NOT isolated waiting',
         'pair_bytes':'unavailable without per-socket/packet attribution; must not infer from collective size'},
         'workers':mapping,'per_worker':[],'pair_traffic_bytes':None}
 by_rank=defaultdict(list)
 for row in records:by_rank[int(row['rank'])].append(row)
 steps_by_rank=defaultdict(list)
 for row in steps:steps_by_rank[int(row['rank'])].append(row)
 for rank in ranks:
  rr=sorted(by_rank[rank],key=lambda r:number(r,'start_unix_s',0))
  ss=steps_by_rank[rank]
  times=[number(r,'start_unix_s') for r in rr if number(r,'start_unix_s') is not None]
  durations=[number(r,'duration_host_s') for r in rr if number(r,'duration_host_s') is not None]
  sizes=[number(r,'bytes_logical') for r in rr if number(r,'bytes_logical') is not None]
  inter=[max(0,times[i]-times[i-1]) for i in range(1,len(times))]
  elapsed=[number(r,'elapsed_s',0) for r in ss]
  steps_n=[number(r,'step',0) for r in ss]
  total=sum(sizes)
  wall=(max(times)-min(times)) if len(times)>1 else None
  # Sweep intervals of host-side collective submissions; NOT network-active concurrency.
  events=[]
  for row in rr:
   t=number(row,'start_unix_s');end=number(row,'end_unix_s')
   if t is not None and end is not None:events.extend([(t,1),(end,-1)])
  active=0;peak=0
  for t,change in sorted(events,key=lambda v:(v[0],v[1])):
   active+=change;peak=max(peak,active)
  result['per_worker'].append({'rank':rank,'node':identity.get(rank,{}).get('node'),
    'ip':identity.get(rank,{}).get('ip'),'collective_bucket_events':len(rr),
    'logical_bytes_total_logged':int(total),'logical_mib_total_logged':round(total/1048576,5),
    'logical_mib_per_bucket':describe([v/1048576 for v in sizes]),
    'interarrival_seconds':describe(inter),'host_collective_completion_s':describe(durations),
    'observed_bucket_events_per_second':len(rr)/wall if wall and wall>0 else None,
    'max_concurrent_host_futures':peak,
    'logged_step_count':len(ss),'last_logged_step':int(steps_n[-1]) if steps_n else None,
    'step_interarrival_seconds':describe([elapsed[i]-elapsed[i-1] for i in range(1,len(elapsed))]),
    'warning':'Per-worker communication totals refer to collected events only; not inferred peer traffic'})
  if rank==0 and times:
   graph(out/'17_collective_interarrival.png',list(range(1,len(times))),inter,'Collective event index','Host interarrival (s)','E1_a — DDP bucket event spacing (rank 0)')
   graph(out/'18_collective_payload.png',list(range(len(sizes))),[v/1048576 for v in sizes], 'Collective bucket event index','Logical payload (MiB)','E1_a — Logical bucket sizes (rank 0)')
 # NCCL topology logs: observed channel graph, not a physical routing map. May be absent unless NCCL_DEBUG=INFO.
 pattern=re.compile(r'Channel\s+\d+(?:/\d+)?\s*:\s*(\d+)\[\d+\]\s*->\s*(\d+)\[\d+\]\s*(?:\[(send|receive)\])?\s*via\s*(\S+)')
 edges={}
 for f in root.glob('node_*.log'):
  for line in f.read_text(errors='replace').splitlines():
   m=pattern.search(line)
   if m:
    u,v,mode,transport=m.groups(); key=(int(u),int(v),transport)
    edges[key]=edges.get(key,0)+1
 result['observed_nccl_channel_edges']=[{'src_rank':u,'dst_rank':v,'transport':tr,
    'src_node':identity.get(u,{}).get('node'),'dst_node':identity.get(v,{}).get('node'),
    'occurrences':count,'bytes':None,'note':'NCCL logged channel route; occurrences are log lines, not actual transfers'} for (u,v,tr),count in sorted(edges.items())]
 result['concurrent_flows_note']='Max concurrent host futures is NOT simultaneous physical flows; NCCL tree/ring internal flows need NCCL transport instrumentation'
 with (root/'traffic_demand.json').open('w') as f:json.dump(result,f,indent=2)
 # machine-friendly source/destination edges, no invented bytes.
 with (root/'observed_nccl_paths.csv').open('w',newline='') as f:
  fields=['src_rank','dst_rank','transport','src_node','dst_node','occurrences','bytes','note'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows(result['observed_nccl_channel_edges'])
 with (root/'sync_wait_summary.csv').open('w',newline='') as f:
  fields=['rank','event','events','median_duration_s','p95_duration_s','max_duration_s'];w=csv.DictWriter(f,fields);w.writeheader()
  grp=defaultdict(list)
  for r in barriers:
   v=number(r,'duration_s')
   if v is not None:grp[(r['rank'],r['event'])].append(v)
  for (rank,event),dur in sorted(grp.items()):w.writerow(dict(rank=rank,event=event,events=len(dur),median_duration_s=percentile(dur,50),p95_duration_s=percentile(dur,95),max_duration_s=max(dur)))
 print('Saved traffic_demand.json, observed_nccl_paths.csv, sync_wait_summary.csv; extra figures in',out)
if __name__=='__main__':main()
