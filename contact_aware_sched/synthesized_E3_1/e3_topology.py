"""Verified NCCL ring-channel union including reverse TCP ACK paths.

Incomplete/inconsistent logs fall back to all directed worker pairs.
"""
import re
from pathlib import Path

_CHANNEL=re.compile(r'NCCL INFO Channel\s+(\d+)/(\d+)\s*:\s*((?:\d+\s*)+)$')

def all_edges(world=4):
    return [(i,j) for i in range(world) for j in range(world) if i!=j]

def ring_edges(logs,world=4):
    tables=[]
    for log in logs:
        if not log or 'NCCL INFO' not in log:return None
        table={}
        counts=set()
        for line in log.splitlines():
            m=_CHANNEL.search(line.strip())
            if not m:continue
            idx,total=int(m[1]),int(m[2])
            order=list(map(int,m[3].split()))
            if total<1 or not 0<=idx<total or sorted(order)!=list(range(world)):return None
            edges=frozenset((a,b) for i,a in enumerate(order)
                            for b in (order[i-1],order[(i+1)%world]))
            if idx in table and table[idx]!=edges:return None
            table[idx]=edges
            counts.add(total)
        if len(counts)>1:return None
        if counts:tables.append((next(iter(counts)),table))
    if not tables or len({n for n,_ in tables})!=1:return None
    total=tables[0][0]
    complete=next((t for n,t in tables if set(t)==set(range(total))),None)
    if complete is None:return None
    if any(complete.get(idx)!=edges for _,t in tables for idx,edges in t.items()):return None
    return sorted(set().union(*complete.values()))

def resolve_topology(output,policy,world=4):
    if policy not in ('ring_auto','full_mesh'):raise ValueError('invalid topology policy')
    logs=[]
    for rank in range(world):
        try:logs.append((Path(output)/f'nccl_rank{rank}.log').read_text(errors='replace'))
        except OSError:logs.append('')
    detected=ring_edges(logs,world)
    required=detected if policy=='ring_auto' and detected else all_edges(world)
    return dict(requested=policy,
        effective='verified_ring_union' if policy=='ring_auto' and detected else 'full_mesh',
        fallback=policy=='ring_auto' and detected is None,
        algorithm='Ring',detected_edges=detected,required_edges=required)
