#!/usr/bin/env python3
import argparse,csv,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--window-ms',type=int,default=1000)
a=p.parse_args();root=Path(a.run_dir)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
path=root/'tcp_rank_pair_windows.csv'
if not path.exists():raise SystemExit('TCP windows missing')
series=collections.defaultdict(list)
with path.open(newline='') as f:
 for r in csv.DictReader(f):
  if int(r['window_ms'])==a.window_ms:
   series[(r['src_rank'],r['dst_rank'])].append((float(r['window_start_unix_s']),float(r['tcp_payload_mbps'])))
if not series:raise SystemExit(0)
t0=min(t for s in series.values() for t,_ in s)
fig,ax=plt.subplots(figsize=(11,5))
for (src,dst),vals in sorted(series.items()):
 ax.step([t-t0 for t,_ in vals],[v for _,v in vals],where='post',label=f'{src}→{dst}')
ax.set(xlabel='Seconds since first sample bin',ylabel='TCP payload Mbps (sample-end bins)',title='Rootless TCP socket counter increments (not packet timestamps)')
ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.25);fig.tight_layout()
out=root/'plots';out.mkdir(exist_ok=True)
fig.savefig(out/f'tcp_pair_{a.window_ms}ms.png',dpi=170)
