#!/usr/bin/env python3
"""Generate a 5-hour deterministic E2 synthetic contact plan by repeating the original
one-hour schedule five times. Twelve symmetric directed edges, no orbital ephemeris.
"""

import csv,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
PERIOD_S=3600
REPEATS=5
HORIZON_S=PERIOD_S*REPEATS
MAX_ORIGINAL_MBPS=120
MAX_TARGET_BPS=9_800_000_000
SCENARIO_ID='synthetic-ddp-demo-v1-scaled-9p8gbps'
# start,end,rate Mbps,delay ms. Entire timeline must be covered.
A=[(0,360,100),(360,540,20),(540,660,0),(660,1080,100),(1080,1260,30),(1260,1440,0),(1440,1980,120),(1980,2160,15),(2160,2340,0),(2340,2880,80),(2880,3060,25),(3060,3180,0),(3180,3600,100)]
B=[(0,300,70),(300,660,0),(660,900,80),(900,1440,0),(1440,1650,70),(1650,2340,0),(2340,2520,70),(2520,3180,0),(3180,3360,80),(3360,3600,0)]
C=[(0,360,90),(360,660,0),(660,1080,90),(1080,1440,0),(1440,1980,100),(1980,2340,0),(2340,2880,70),(2880,3180,0),(3180,3600,90)]
plans={(1,2):(A,8),(1,3):(B,12),(1,4):(C,10),(2,3):(B,4),(2,4):(C,2),(3,4):(B,2)}
p=ROOT/'traces/directed_trace_4workers.csv'
with p.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scenario_id','src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
 for (a,b),(segments,delay) in sorted(plans.items()):
  assert segments[0][0]==0 and segments[-1][1]==3600
  assert all(x[1]==y[0] for x,y in zip(segments,segments[1:]))
  for u,v in [(a,b),(b,a)]:
   for start,end,rate in segments:
    for cycle in range(REPEATS):
     offset=cycle*PERIOD_S
     w.writerow([SCENARIO_ID,f'SAT-DEMO-{u:03d}',f'SAT-DEMO-{v:03d}',start+offset,end+offset,int(rate>0),int(round(rate*MAX_TARGET_BPS/MAX_ORIGINAL_MBPS)),delay if rate else 0])
print(f'{p} ({HORIZON_S}s / {HORIZON_S/3600:g}h, {REPEATS} repeated cycles)')
