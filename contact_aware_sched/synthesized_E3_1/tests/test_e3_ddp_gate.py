#!/usr/bin/env python3
"""Offline regression checks for DDP-only contact scheduling."""
import ast
import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))

def load_policy():
    spec=importlib.util.spec_from_file_location('ddp_schedule_test',BASE/'ddp'/'schedule.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ContactScheduler

class DdpGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.trace=self.root/'trace.csv'
        with self.trace.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
            w.writeheader()
            sats=[f'SAT-DEMO-{i+1:03d}' for i in range(4)]
            for a in sats:
                for b in sats:
                    if a!=b:
                        w.writerow(dict(src=a,dst=b,t_start_s=0,t_end_s=60,
                                        available=1,rate_bps=10_000_000_000,prop_delay_ms=2))
    def test_ddp_default_launch_guard_is_positive(self):
        Cls=load_policy()
        sched=Cls(self.trace,time.time(),30,0,'ddp',self.root/'rank0',mode='aware')
        self.assertGreater(sched.launch_guard,0.0)
        sched.close()

    def test_ddp_rejects_unknown_modes(self):
        Cls=load_policy()
        for mode in ('proactive',):
            with self.assertRaises(ValueError):
                Cls(self.trace,time.time(),30,0,'ddp',
                    self.root/('reject_'+mode),mode=mode)

    def test_other_methods_keep_default_lead(self):
        from e3_schedule import ContactScheduler
        sched=ContactScheduler(self.trace,time.time(),30,0,'diloco64',self.root/'other',mode='off')
        self.assertEqual(sched.rendezvous_lead_s,0.2)
        sched.close()

    def test_explicit_override_supported(self):
        Cls=load_policy()
        sched=Cls(self.trace,time.time(),30,0,'ddp',self.root/'override',mode='aware',rendezvous_lead_s=.05)
        self.assertEqual(sched.rendezvous_lead_s,.05)
        sched.close()

    def test_native_ddp_gate_before_forward(self):
        text=(BASE/'nccl_train.py').read_text()
        ast.parse(text)
        gate=text.index("scheduler.gate('collective',step,logical")
        forward=text.index("logits=model(x);loss=criterion(logits,y)")
        backward=text.index("scaler.scale(loss).backward()")
        opt=text.index("scaler.step(opt)")
        self.assertLess(gate,forward)
        self.assertLess(forward,backward)
        self.assertLess(backward,opt)
        self.assertIn('DistributedDataParallel(',text)
        self.assertIn('broadcast_buffers=False',text)
        self.assertIn('scheduler.finish_operation(',text)

    def test_four_worker_rendezvous_without_fixed_delay(self):
        Cls=load_policy()
        # Future contact start permits all workers to rendezvous before T0.
        start=time.time()+.5
        out=self.root/'shared'
        scheds=[Cls(self.trace,start,30,rank,'ddp',out,mode='aware',
                   margin=.1,min_transfer=.01,factor=1) for rank in range(4)]
        statuses=[None]*4
        def do_gate(rank):
            statuses[rank]=scheds[rank].gate('collective',0,1024)
        with patch.dict(os.environ,{'E3_RENDEZVOUS_LEAD_S':'0'}):
            threads=[threading.Thread(target=do_gate,args=(i,)) for i in range(4)]
            for t in threads:t.start()
            for t in threads:t.join(timeout=5)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertTrue(all(status and status[0] for status in statuses))
        for s in scheds:s.close()

if __name__=='__main__':unittest.main()
