#!/usr/bin/env python3
"""Directional TCP payload estimates from cumulative ss counters; never packet bytes.

Each stream delta is assigned to its known local source rank. The destination rank
is assigned only if a remote rank is uniquely identified by worker IP or the
reverse four-tuple is observed. Bytes_received are NOT added to sender totals.
"""
import argparse,collections,csv,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--windows-ms',default='100,1000')
a=p.parse_args();root=Path(a.run_dir)
workers=json.loads((root/'worker_map.json').read_text())['workers']
by_ip=collections.defaultdict(set)
for w in workers:by_ip[w['ip']].add(int(w['rank']))
windows=sorted(set(int(x) for x in a.windows_ms.split(',')))
assert windows and min(windows)>0

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
samples=[]
for f in root.glob('tcp_socket_samples_*.csv'):samples+=read(f)
# A local 4tuple plus rank identifies a flow until closure/reuse. Counter resets are flagged.
streams=collections.defaultdict(list)
for r in samples:
    key=(r['host'],r['rank'],r['pid'],r['src_ip'],r['src_port'],r['dst_ip'],r['dst_port'])
    streams[key].append(r)
# Four-tuple reverse lookup gives rank identity if both sides sampled, even with multi-GPU hosts.
endpoints=collections.defaultdict(set)
for key in streams:
    host,rank,pid,ip,port,peer,pport=key
    endpoints[(ip,port,peer,pport)].add(int(rank))

def rank_dst(key):
    _,src_rank,_,src,sp,dst,dp=key
    opposite=endpoints.get((dst,dp,src,sp),set())
    if len(opposite)==1:return str(next(iter(opposite))),'reverse_socket'
    choices=by_ip.get(dst,set())
    if len(choices)==1:return str(next(iter(choices))),'unique_host_ip'
    return 'unknown','not_attributable'

out=[];totals=collections.defaultdict(lambda:dict(bytes=0,intervals=0,first=math.inf,last=0,methods=set()))
invalid=0;missing=0;first_samples=0
for key,rows in streams.items():
    rows.sort(key=lambda r:float(r['unix_time_s']))
    last=None
    for r in rows:
        if r['bytes_sent']=='':missing+=1;last=None;continue
        t=float(r['unix_time_s']);n=int(r['bytes_sent'])
        if last is None:
            first_samples+=1;last=(t,n);continue
        pt,pn=last;last=(t,n)
        if n<pn or t<=pt:invalid+=1;continue
        delta=n-pn
        if not delta:continue
        dst,method=rank_dst(key)
        src=key[1]
        # Put volume at observation end; timestamp bounds describe true uncertainty.
        obj={'src_rank':src,'dst_rank':dst,'src_ip':key[3],'dst_ip':key[5],
             'src_port':key[4],'dst_port':key[6], 'start_unix_s':round(pt,6),
             'end_unix_s':round(t,6),'interval_s':round(t-pt,6),
             'tcp_payload_delta_bytes':delta,'rate_mbps':round(8*delta/(t-pt)/1e6,6),
             'attribution_method':method,'observation':'sampled_tcp_bytes_sent_delta'}
        out.append(obj)
        agg=totals[(src,dst)];agg['bytes']+=delta;agg['intervals']+=1
        agg['first']=min(agg['first'],pt);agg['last']=max(agg['last'],t);agg['methods'].add(method)

names=['src_rank','dst_rank','src_ip','dst_ip','src_port','dst_port','start_unix_s','end_unix_s','interval_s','tcp_payload_delta_bytes','rate_mbps','attribution_method','observation']
with (root/'tcp_rank_pair_intervals.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,names);w.writeheader();w.writerows(sorted(out,key=lambda x:x['end_unix_s']))
with (root/'tcp_rank_pair_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,['src_rank','dst_rank','tcp_payload_observed_delta_bytes','intervals','first_unix_s','last_unix_s','attribution_methods']);w.writeheader()
    for (s,d),v in sorted(totals.items(),key=lambda p:(int(p[0][0]),p[0][1])):
        w.writerow(dict(src_rank=s,dst_rank=d,tcp_payload_observed_delta_bytes=v['bytes'],intervals=v['intervals'],
                        first_unix_s=v['first'],last_unix_s=v['last'],attribution_methods=';'.join(sorted(v['methods']))))
# Do not fabricate time resolution finer than the observation interval. Assign each
# delta to its ending bin, retaining original interval start/end for uncertainty.
with (root/'tcp_rank_pair_windows.csv').open('w',newline='') as f:
    fields=['window_ms','window_start_unix_s','window_end_unix_s','src_rank','dst_rank','tcp_payload_delta_bytes','tcp_payload_mbps','method']
    w=csv.DictWriter(f,fields);w.writeheader()
    bins=collections.Counter()
    for v in out:
        for ms in windows:
            binid=math.floor(v['end_unix_s']*1000/ms)
            bins[(ms,binid,v['src_rank'],v['dst_rank'])]+=v['tcp_payload_delta_bytes']
    for (ms,bi,s,d),n in sorted(bins.items()):
        w.writerow(dict(window_ms=ms,window_start_unix_s=round(bi*ms/1000,6),
            window_end_unix_s=round((bi+1)*ms/1000,6),src_rank=s,dst_rank=d,
            tcp_payload_delta_bytes=n,tcp_payload_mbps=round(8*n/(ms*1000),6),method='end_sample_bin_not_packet_timestamp'))
# NIC is an independent, node-total L2-like cross-check, not traffic attributed to training.
with (root/'tcp_nic_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,['host','samples','nic_tx_delta_bytes','nic_rx_delta_bytes','mean_sample_spacing_s','max_sample_spacing_s']);w.writeheader()
    for fn in sorted(root.glob('tcp_monitor_samples_*.csv')):
        rr=read(fn);times=[float(r['unix_time_s']) for r in rr]
        def diff(k):
            nums=[int(r[k]) for r in rr if r[k]!='']
            return nums[-1]-nums[0] if len(nums)>1 and nums[-1]>=nums[0] else ''
        gaps=[b-a for a,b in zip(times,times[1:])]
        w.writerow(dict(host=fn.stem.removeprefix('tcp_monitor_samples_'),samples=len(rr),
          nic_tx_delta_bytes=diff('nic_tx_bytes'),nic_rx_delta_bytes=diff('nic_rx_bytes'),
          mean_sample_spacing_s=round(statistics.mean(gaps),5) if gaps else '',
          max_sample_spacing_s=round(max(gaps),5) if gaps else ''))
notes={'measurement_scope':'rootless ss TCP_INFO cumulative bytes_sent counter differences',
 'not_packet_capture':True,'not_physical_wire_bytes':True,'not_exact_nccl_bytes':True,
 'time_resolution':'observations are sampled; an interval delta can occur anywhere between its start/end; windows use end-of-observation bins',
 'cannot_detect':'connection bytes transferred before first sample, after last sample, or fully between samples; RDMA, GPU P2P, and SHM; TCP header overhead',
 'rate_uncertainty':'bin-based Mbps can be misleading for intervals larger than bins; prefer interval_s and rate_mbps',
 'number_of_workers':len(workers),'streams_seen':len(streams),'intervals_with_positive_bytes':len(out),
 'first_samples_unmeasured':first_samples,'missing_bytes_sent_samples':missing,'counter_resets_or_bad_timestamps':invalid,
 'fully_attributed_intervals':sum(v['dst_rank']!='unknown' for v in out),
 'unattributed_intervals':sum(v['dst_rank']=='unknown' for v in out)}
(root/'tcp_measurement_notes.json').write_text(json.dumps(notes,indent=2))
print('TCP socket streams:',len(streams),'positive-byte intervals:',len(out),'unknown destinations:',notes['unattributed_intervals'])
