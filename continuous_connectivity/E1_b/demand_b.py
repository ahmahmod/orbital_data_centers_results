#!/usr/bin/env python3
"""Summarize application-level directional messages, and create E1_b communication figures.
Do NOT equate logical payload bytes with TCP/NIC physical bytes.
"""
import argparse,csv,json,collections,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):
    if not p.exists():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))
def fnum(r,key,default=0):
    try:return float(r[key])
    except (ValueError,KeyError,TypeError):return default

def plot(out,filename,records,event,title,y='payload_bytes_logical',scale=1048576):
    rr=[r for r in records if r.get('event')==event]
    if not rr:return
    fig,ax=plt.subplots(figsize=(9,4))
    by=collections.defaultdict(list)
    for r in rr:by[int(r['rank'])].append(r)
    for rank,rows in sorted(by.items()):
        rows.sort(key=lambda z:fnum(z,'elapsed_s'))
        ax.plot([fnum(z,'elapsed_s') for z in rows],[fnum(z,y)/scale for z in rows],label=f'rank {rank}',marker='.',markersize=3)
    ax.set(xlabel='Training elapsed (s)',ylabel='Logical payload (MiB)' if y=='payload_bytes_logical' else y,title=title)
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/filename,dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();root=Path(a.run_dir)
    out=root/'plots';out.mkdir(exist_ok=True)
    files=sorted(root.glob('messages_rank*.csv'));records=[]
    for path in files:records.extend(read(path))
    counts=collections.Counter(r.get('event') for r in records)
    method=next((r['method'] for r in records if r.get('method')),None)
    pair=collections.defaultdict(lambda:dict(messages=0,logical_bytes=0,first_unix=None,last_unix=None,versions=[]))
    for r in records:
        if r.get('event')!='send_complete':continue
        key=(r['src_rank'],r['dst_rank']);item=pair[key];item['messages']+=1
        item['logical_bytes']+=int(fnum(r,'payload_bytes_logical'))
        ts=fnum(r,'unix_time_s');item['first_unix']=ts if item['first_unix'] is None else min(ts,item['first_unix'])
        item['last_unix']=ts if item['last_unix'] is None else max(ts,item['last_unix'])
    pairs=[dict(src_rank=k[0],dst_rank=k[1],**v) for k,v in sorted(pair.items())]
    with (root/'p2p_pair_summary.csv').open('w',newline='') as fh:
        cols=['src_rank','dst_rank','messages','logical_bytes','first_unix','last_unix']
        w=csv.DictWriter(fh,fieldnames=cols);w.writeheader()
        for r in pairs:w.writerow({k:r.get(k) for k in cols})
    # Keep method-specific totals separate from the inherited DDP-oriented collective fields.
    result={'experiment':'E1_b','method':method,'events':dict(counts),'directional_p2p':pairs,
        'completed_p2p_logical_bytes':sum(p['logical_bytes'] for p in pairs),
        'outstanding_unconfirmed':counts['send_outstanding_end'],
        'p2p_note':'Only completed sends counted. Messages are full model snapshots, not gradient all-reduce.',
        'diloco_note':'DiLoCo outer synchronization is an all-reduce; exact per-rank network paths/physical bytes not inferred.',
        'transport_note':'Use E1_a_new TCP socket and NIC analyses for observed transport (with sampling limitations).'}
    (root/'traffic_demand.json').write_text(json.dumps(result,indent=2))
    plot(out,'17_diloco_outer_transfers.png',records,'outer_allreduce','DiLoCo outer delta all-reduce logical payload')
    plot(out,'18_gossip_completed_sends.png',records,'send_complete','Gossip directed completed sends (logical payload)')
    plot(out,'19_gossip_receive_staleness.png',records,'receive_mix','Gossip received version vs training time',y='received_version',scale=1)
    print('Saved E1_b traffic_demand.json, p2p_pair_summary.csv, communication figures')
if __name__=='__main__':main()
