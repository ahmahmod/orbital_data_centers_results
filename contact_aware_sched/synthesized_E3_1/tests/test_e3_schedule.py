#!/usr/bin/env python3
"""Offline, no CUDA, no OAR: scheduler safety and paired-run invariants."""
import csv,sys,tempfile,threading,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_schedule import ContactScheduler

class ScheduleTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.trace=self.root/'plan.csv'
        fields=['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms']
        with self.trace.open('w',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=fields);w.writeheader()
            sats=[f'SAT-DEMO-{i+1:03d}' for i in range(4)]
            for a in sats:
                for b in sats:
                    if a==b:continue
                    for t0,t1,avail in [(0,5,1),(5,10,0),(10,40,1)]:
                        w.writerow(dict(src=a,dst=b,t_start_s=t0,t_end_s=t1,
                                     available=avail,rate_bps=1_000_000_000 if avail else 0,
                                     prop_delay_ms=4))
    def tearDown(self):self.tmp.cleanup()
    def make(self,rank=0,mode='off',start=None,method='gossip8',output=None):
        return ContactScheduler(self.trace,start or time.time(),40,rank,method,output or self.root/f'run_{mode}_{rank}',
             mode=mode,margin=.1,min_transfer=.01,factor=1.0)
    def test_off_is_immediate_and_log_is_complete(self):
        plan=self.make(mode='off')
        before=time.time()
        ok,op=plan.gate('peer',8,1024,peer=1,version=8)
        self.assertTrue(ok);self.assertLess(time.time()-before,.2)
        plan.complete(op,8,1024,peer=1)
        plan.close()
        with (self.root/'run_off_0/e3_scheduler_decisions_rank0.csv').open() as f:
            records=list(csv.DictReader(f))
        self.assertEqual(len(records),2)
        self.assertEqual(records[0]['scheduler_action'],'start')
        self.assertNotEqual(records[0]['comm_start_unix'],'')
        self.assertEqual(records[-1]['communication_status'],'completed')
    def test_aware_no_future_window_does_not_send(self):
        plan=self.make(mode='aware',start=time.time()-43)
        ok,_=plan.gate('peer',8,1024,peer=1,version=8)
        self.assertFalse(ok)
        plan.close()
    def test_all_worker_rendezvous(self):
        start=time.time()+1
        schedulers=[self.make(rank=i,mode='aware',start=start,method='ddp',output=self.root/'shared_run') for i in range(4)]
        outcomes=[None]*4
        def execute(i):
            outcomes[i]=schedulers[i].gate('collective',0,1024)
        threads=[threading.Thread(target=execute,args=(i,)) for i in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=10)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertTrue(all(o[0] for o in outcomes))
        for s in schedulers:s.close()
    def test_contact_window_clipped_to_short_horizon(self):
        # Regression: a 3-minute pilot ends before the first 300-second contact.
        # The contact oracle must retain the portion within the run horizon.
        plan=ContactScheduler(self.trace,time.time(),3,0,'gossip64',
              self.root/'shortpilot',mode='aware',margin=.1,min_transfer=.01)
        self.assertEqual([(a,b) for a,b,*_ in plan.windows(plan.required('collective'))],
                         [(0.0,3.0)])
        plan.close()

    def test_only_declared_links(self):
        plan=self.make(mode='off')
        self.assertEqual(len(plan.required('collective')),12)
        self.assertEqual(len(plan.required('flower')),6)
        self.assertEqual(len(plan.required('peer',3)),1)
        w=plan.windows(plan.required('collective'))
        self.assertEqual([(a,b) for a,b,*_ in w],[(0,5),(10,40)])
        plan.close()

if __name__=='__main__':unittest.main()
