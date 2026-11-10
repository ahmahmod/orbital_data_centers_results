#!/usr/bin/env python3
"""Normalize E1_a DDP bucket records to an E1_b-comparable logical event schema.

A DDP collective is NOT an individually directed peer message. Never fill a
src_rank/dst_rank pair or equate input bucket size with physical TCP bytes.
"""
import argparse,csv,json
from pathlib import Path

FIELDS=['experiment','method','rank','step','epoch','unix_time_s','elapsed_s','event',
        'src_rank','dst_rank','src_ip','dst_ip','payload_bytes_logical','duration_host_s',
        'status','collective_membership','measurement_scope']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_dir');a=ap.parse_args();root=Path(a.run_dir)
    result=root/'communication_events.csv';n=0;bytes_by_rank={}
    with result.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader()
        for fn in sorted(root.glob('collectives_rank*.csv')):
            with fn.open(newline='') as f:
                for r in csv.DictReader(f):
                    if r.get('status')!='ok':continue
                    rank=int(r['rank']);size=int(r['bytes_logical']);n+=1
                    bytes_by_rank[rank]=bytes_by_rank.get(rank,0)+size
                    writer.writerow(dict(experiment='E1_a',method='ddp',rank=rank,step=r['step'],epoch='',
                        unix_time_s=r['end_unix_s'],elapsed_s='',event='allreduce_complete',
                        src_rank='',dst_rank='',src_ip=r.get('src_ip',''),dst_ip='',
                        payload_bytes_logical=size,duration_host_s=r.get('duration_host_s',''),status='ok',
                        collective_membership='all_ranks',
                        measurement_scope='one rank submitting a logical gradient bucket; NOT a unicast or network transfer'))
    (root/'communication_events_summary.json').write_text(json.dumps(dict(method='ddp',events=n,
       logical_bucket_bytes_by_rank=bytes_by_rank,
       note='Input gradient bucket volume per rank. Sum across ranks is NOT physical network traffic. '
            'Only bucket events explicitly traced in collectives_rank*.csv are included.'),indent=2))
    print(f'Normalized {n} DDP bucket events to {result}')
if __name__=='__main__':main()
