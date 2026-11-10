#!/usr/bin/env python3
"""Monitor completeness; a successful DDP run does not guarantee TCP visibility."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--expected-hosts',type=int,required=True)
a=p.parse_args();root=Path(a.run_dir)
rows=[]
for path in sorted(root.glob('tcp_monitor_samples_*.csv')):
    with path.open(newline='') as f:rr=list(csv.DictReader(f))
    rows.append({'host':path.stem.removeprefix('tcp_monitor_samples_'),'samples':len(rr),
        'errors':sum(int(r['ss_returncode'])!=0 for r in rr),'owned_socket_samples':sum(int(r['owned_connections']) for r in rr),
        'connections_seen_total':sum(int(r['connections_seen']) for r in rr),
        'map_available_samples':sum(int(r['worker_map_available']) for r in rr),
        'finished':(root/f"monitor_finished_{path.stem.removeprefix('tcp_monitor_samples_')}").exists()})
result={'expected_hosts':a.expected_hosts,'observed_hosts':len(rows),'hosts':rows,
        'monitor_completeness_ok':len(rows)==a.expected_hosts and all(r['samples']>0 and r['errors']==0 and r['finished'] for r in rows),
        'rank_socket_visibility_ok':any(r['owned_socket_samples']>0 for r in rows),
        'directional_tcp_byte_intervals_observed':False,
        'traffic_visibility_warning':'zero owner observations is not evidence of zero NCCL traffic; inspect ss -p permissions, transport, and worker mapping'}
summary=root/'tcp_rank_pair_intervals.csv'
if summary.exists():
    with summary.open(newline='') as f:result['directional_tcp_byte_intervals_observed']=any(int(row['tcp_payload_delta_bytes'])>0 for row in csv.DictReader(f))
(root/'socket_monitor_health.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if not result['monitor_completeness_ok']:raise SystemExit(1)
