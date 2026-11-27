"""CPU-only regression tests with real TCP sockets (no GPU or tc privileges).

Loopback tests exercise protocol ordering and predicted contact scheduling,
not kernel netem loss/retransmission performance on the Grid'5000 network.
"""
import csv
import socket
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from communication.contact_plan import LinkPlan, RingConnectivity
from communication.transport import RingTransport
from communication.ring_allreduce import RingAllReduce


def segments(duration,windows,rate=1e9,delay=.001):
    bounds=sorted({0.,float(duration)} |
                  {x for a,b in windows for x in (a,b) if 0<x<duration})
    return [(a,b,int(any(x<=(a+b)/2<y for x,y in windows)),rate,delay)
            for a,b in zip(bounds,bounds[1:])]


def trace(duration,kind):
    sats=[f'SAT-DEMO-{i+1:03d}' for i in range(4)]
    edges={}
    for a in range(4):
        for b in range(4):
            if a==b:continue
            if kind=='all_up':windows=[(0,duration)]
            elif kind=='staggered':
                index=next((i for i in range(4) if {a,b}=={i,(i+1)%4}),None)
                windows=[] if index is None else [
                    (k*.8+index*.2,min(duration,k*.8+index*.2+.17))
                    for k in range(int(duration/.8)+1)
                    if k*.8+index*.2<duration]
            else:windows=[]
            edges[sats[a],sats[b]]=segments(duration,windows)
    return sats,edges


class PlanTests(unittest.TestCase):
    def test_forward_and_ack_overlap(self):
        plan=LinkPlan(segments(20,[(0,10)]),segments(20,[(5,15)]),20,
                      margin=.2,open_guard=.1,min_transfer=.01)
        self.assertEqual(len(plan.windows),1)
        start,end,estimate=plan.next_slot(0,1024)
        self.assertAlmostEqual(start,5.1)
        self.assertEqual(end,10)
        self.assertGreater(estimate,0)
        self.assertIsNone(plan.next_slot(9.99,1024))

    def test_rate_and_size_budget(self):
        plan=LinkPlan(segments(2,[(0,2)],rate=1e6),
                      segments(2,[(0,2)],rate=1e6),2,
                      margin=.1,open_guard=.01,factor=2)
        self.assertIsNotNone(plan.next_slot(0,1024))
        self.assertIsNone(plan.next_slot(0,1048576))


    def test_adaptive_credit_capacity(self):
        plan=LinkPlan(segments(2,[(0,2)],rate=1e9,delay=.01),
                      segments(2,[(0,2)],rate=1e9,delay=.01),2,
                      margin=.1,open_guard=.01,factor=2,min_transfer=.001)
        payload=[1048576]*32
        self.assertGreater(plan.credit_limit(.1,payload,32),1)
        self.assertLess(plan.credit_limit(.1,payload,32),32)
        self.assertEqual(plan.credit_limit(1.99,payload,32),0)
        self.assertEqual(plan.credit_limit(0,payload,32),0)
        self.assertLessEqual(plan.credit_limit(.1,payload,2),2)

    def test_partial_connectivity_classification(self):
        sats,edges=trace(8,'staggered')
        plan=RingConnectivity(edges,sats,8)
        self.assertEqual(plan.classify(.02,.04),'partial')
        self.assertEqual(plan.classify(.02,.18),'crossing')
        self.assertEqual(plan.classify(.18,.19),'none')
        self.assertEqual(plan.classify(7.99,8.01),'outside')


class RingTests(unittest.TestCase):
    def exercise(self,mode,kind,duration,ops):
        sats,edges=trace(duration,kind)
        peers=['127.0.0.1','127.0.0.2','127.0.0.3','127.0.0.4']
        with socket.socket() as sock:
            sock.bind((peers[0],0))
            port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            epoch=[None]
            def set_start():
                epoch[0]=time.time()+.15
            connected=threading.Barrier(4,action=set_start,timeout=8)
            def worker(rank):
                sched=SimpleNamespace(mode=mode,start=time.time(),end=time.time()+duration,
                                      sats=sats,edges=edges,output=root)
                # Exact duration is shared in the real trainer too.
                sched.end=sched.start+duration
                tr=RingTransport(rank,peers,port,'test-run',sched,frame_bytes=32,
                    margin=.01,factor=1,open_guard=.01,min_transfer=.001,
                    connect_timeout=5,io_timeout=5,drain_guard=.05,drain_timeout=5)
                try:
                    tr.connect()
                    connected.wait()
                    tr.set_epoch(epoch[0],duration)
                    while time.time()<epoch[0]:time.sleep(.002)
                    reducer=RingAllReduce(tr)
                    result=[]
                    for step,explicit_stop in enumerate(ops):
                        vector=np.arange(73,dtype=np.float32)+100*rank+step
                        stop=(rank==2) if explicit_stop=='one_rank' else explicit_stop
                        mean,ended=reducer.reduce_mean(vector,step,stop)
                        if not ended:
                            np.testing.assert_allclose(mean,np.arange(73,dtype=np.float32)+150+step)
                        else:self.assertIsNone(mean)
                        result.append(ended)
                    return result
                finally:tr.close()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(worker,rank) for rank in range(4)]
                outcomes=[f.result(timeout=duration+15) for f in futures]
            records=[]
            for rank in range(4):
                with (root/f'e3_ring_transfers_rank{rank}.csv').open() as fp:
                    records+=list(csv.DictReader(fp))
            tx=[r for r in records if r['direction']=='tx']
            self.assertTrue(tx)
            self.assertTrue(all(r['status']=='completed' for r in records))
            return outcomes,tx

    def test_off_ignores_contacts_and_global_stop_vote(self):
        results,tx=self.exercise('off','all_down',4,[False,False,'one_rank'])
        self.assertEqual(results,[[False,False,True]]*4)
        self.assertTrue(all(float(r['contact_wait_s'])==0 for r in tx))
        self.assertTrue(any(int(r['outside_contact_frames'])>0 for r in tx))
        self.assertTrue(any(int(r['max_outstanding_frames'])>1 for r in tx))
        self.assertTrue(all(int(r['post_horizon_frames'])==0 for r in tx))

    def test_aware_without_any_common_four_worker_window(self):
        results,tx=self.exercise('aware','staggered',8,[False])
        self.assertEqual(results,[[False]]*4)
        self.assertTrue(any(float(r['contact_wait_s'])>.01 for r in tx))
        self.assertTrue(all(int(r['outside_contact_frames'])==0 for r in tx))
        self.assertTrue(all(int(r['post_horizon_frames'])==0 for r in tx))
        # 4 workers x 6 exchanges = 24 TX chunks per collective.
        self.assertEqual(len(tx),24)
        self.assertTrue(all(int(r['frames'])>1 for r in tx))
        self.assertTrue(any(int(r['max_outstanding_frames'])>1 for r in tx))
        self.assertTrue(any(int(r['partial_contained_bytes'])>0 for r in tx))
        for row in tx:
            self.assertEqual(int(row['ack_payload_in_horizon']),sum(int(row[k]) for k in ('partial_contained_bytes','full_contained_bytes','crossing_bytes','none_contained_bytes')))

    def test_horizon_drain_and_common_terminal_vote(self):
        results,tx=self.exercise('aware','all_down',.3,[False,None])
        self.assertEqual(results,[[False,True]]*4)
        self.assertTrue(all(int(r['outside_contact_frames'])==0 for r in tx))
        self.assertTrue(any(int(r['post_horizon_frames'])>0 for r in tx))


if __name__=='__main__':unittest.main()
