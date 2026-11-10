#!/usr/bin/env python3
"""Document NCCL observability: topology debug logs do not quantify transfer bytes."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();root=Path(a.run_dir)
logs=list(root.glob('node_*.log'))
text='\n'.join(x.read_text(errors='replace')[:4_000_000] for x in logs)
vers=sorted(set(re.findall(r'NCCL version\s+([^\s]+)',text)))
network_lines=[line[:350] for line in text.splitlines() if ('via NET/' in line or 'via P2P/' in line or 'via SHM/' in line)][:150]
report={'nccl_versions_in_logs':vers,'observed_transport_log_lines':network_lines,
 'collective_logical_bytes_available':bool(list(root.glob('collectives_rank*.csv'))),
 'nccl_directional_transport_bytes_available':False,
 'reason':'NCCL_DEBUG topology/transport messages do not contain completed isend/irecv byte counters. No compiled NCCL profiler or network plugin is installed by this package.',
 'future_extension':'Build and validate an NCCL-version-compatible profiler/network-plugin instrumenting peer identity and per-send completion; collect separately from ss counters.',
 'tcp_observability':'Only TCP_INFO ss byte counters are observed; RDMA / SHM / P2P are not recorded by ss.'}
(root/'nccl_observability.json').write_text(json.dumps(report,indent=2))
print('Saved NCCL observability diagnostic:',root/'nccl_observability.json')
