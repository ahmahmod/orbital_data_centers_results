"""CPU tests for topology verification, recent estimates and frontend control."""
import csv
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_broker import Broker, ControlClient
from e3_topology import ring_edges, resolve_topology
from e3_batch_control import BatchScheduler


def make_trace(path, duration=300):
    with path.open('w') as fp:
        w=csv.writer(fp)
        w.writerow(['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
        for i in range(1,5):
            for j in range(1,5):
                if i!=j:w.writerow([f'SAT-DEMO-{i:03d}',f'SAT-DEMO-{j:03d}',0,duration,1,1e9,1])


class TopologyTests(unittest.TestCase):
    def test_one_ring_reverse_edges(self):
        logs=['NCCL INFO Channel 00/01 : 0 1 2 3']+['NCCL INFO init']*3
        edges=ring_edges(logs)
        self.assertEqual(len(edges),8)
        self.assertIn((0,3),edges)
        self.assertIn((3,0),edges)
        self.assertNotIn((0,2),edges)

    def test_union_and_fallback(self):
        logs=['NCCL INFO Channel 00/02 : 0 1 2 3\nNCCL INFO Channel 01/02 : 0 2 1 3']+['NCCL INFO init']*3
        self.assertEqual(len(ring_edges(logs)),12)
        self.assertIsNone(ring_edges([logs[0].splitlines()[0]]+logs[1:]))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(resolve_topology(tmp,'ring_auto')['required_edges']),12)


class BrokerTests(unittest.TestCase):
    def test_visibility_and_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=Broker('127.0.0.1','secret',tmp)
            a=ControlClient(server.endpoint,'secret')
            b=ControlClient(server.endpoint,'secret')
            try:
                self.assertIsNone(a.read('nccl/batch/decision.json'))
                b.write('nccl/batch/decision.json',{'run_id':'run','K':32})
                self.assertEqual(a.read('nccl/batch/decision.json')['K'],32)
                self.assertTrue((Path(tmp)/'.e3_control/nccl/batch/decision.json').exists())
                b.write('replay/rank0/SAT-DEMO-002',{'available':1})
                self.assertEqual(a.read('replay/rank0/SAT-DEMO-002')['available'],1)
                self.assertFalse((Path(tmp)/'.e3_control/replay').exists())
            finally:
                a.close();b.close();server.close()

    def test_four_workers_grant_and_drain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trace=root/'trace.csv'
            make_trace(trace,10)
            server=Broker('127.0.0.1','secret',root)
            errors=[];outcomes=[[] for _ in range(4)]
            env=dict(E3_CONTROL_ENDPOINT=server.endpoint,E3_CONTROL_TOKEN='secret',
                E3_CONTROL_POLL_S='.01',E3_RENDEZVOUS_TIMEOUT='3',E3_VERIFY_REPLAY='0',
                E3_MIN_TRANSFER_S='.01',E3_LOCAL_STEP_BUDGET_S='.01',
                E3_GRANT_MAX_OPS='4',E3_CONTACT_OPEN_GUARD_S='.01',
                E3_LAUNCH_GUARD_S='.01')
            try:
                with patch.dict(os.environ,env):
                    t0=time.time()+.2
                    def worker(rank):
                        s=BatchScheduler(trace,t0,10,rank,'ddp',root,mode='aware',
                            margin=.02,factor=1,run_id='broker-test',comm_timeout=1)
                        try:
                            for step in range(5):
                                ok,op=s.gate('collective',step,1024,remaining_ops=5-step)
                                outcomes[rank].append(ok)
                                if not ok:break
                                s.complete(op,step,1024);s.finish_operation(op,step)
                        except Exception as exc:errors.append(repr(exc))
                        finally:s.close()
                    ts=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(4)]
                    for t in ts:t.start()
                    for t in ts:t.join(5)
                    self.assertFalse(any(t.is_alive() for t in ts))
                    self.assertEqual(errors,[])
                    self.assertEqual(outcomes,[[True]*5]*4)
            finally:server.close()


class EstimatorTests(unittest.TestCase):
    def test_stale_outlier_expires_inside_contact(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
                'E3_OBSERVED_MAX_AGE_S':'120','E3_OBSERVED_WINDOW':'32',
                'E3_OBSERVED_QUANTILE':'.9','E3_MIN_TRANSFER_S':'.5'}):
            root=Path(tmp);trace=root/'trace.csv'
            make_trace(trace)
            t0=time.time()
            s=BatchScheduler(trace,t0,300,0,'ddp',root,mode='aware',margin=5,run_id='est')
            try:
                sample=dict(stamp=t0+10,rate=1e9,delay=.001,elapsed=1000,full=True)
                rows=[dict(samples=[sample]) for _ in range(4)]
                plan=s._plan(t0+20,1024,32,rows)
                self.assertIsNotNone(plan)
                self.assertAlmostEqual(plan['start'],t0+130,places=3)
                self.assertLess(plan['estimate'],5)
                self.assertEqual(plan['K'],1)
                self.assertEqual(s._recent(rows,t0+20,4e9,.001,True),(0.,False))
            finally:s.close()


if __name__=='__main__':
    unittest.main()
