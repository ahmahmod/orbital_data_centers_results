"""CPU tests for count agreement, fast path, fixed H and stopping."""
import csv
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3_batch_control import BatchScheduler


class BatchTest(unittest.TestCase):
    def run_case(self, method, mode, windows, count, max_ops=4):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {
            'E3_GRANT_MAX_OPS':str(max_ops),'E3_LOCAL_STEP_BUDGET_S':'.015',
            'E3_OBSERVED_FACTOR':'1.5','E3_MIN_TRANSFER_S':'.015',
            'E3_CONTACT_OPEN_GUARD_S':'.015','E3_LAUNCH_GUARD_S':'.025',
            'E3_CONTROL_POLL_S':'.002','E3_RENDEZVOUS_TIMEOUT':'3',
            'E3_VERIFY_REPLAY':'0'}):
            root=Path(tmp)
            trace=root/'trace.csv'
            with trace.open('w') as fp:
                w=csv.writer(fp)
                w.writerow(['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
                for i in range(1,5):
                    for j in range(1,5):
                        if i!=j:
                            for start,end,up in windows:
                                w.writerow([f'SAT-DEMO-{i:03d}',f'SAT-DEMO-{j:03d}',start,end,up,1e9,1])
            t0=time.time()+.10
            h=int(''.join(c for c in method if c.isdigit()) or '1')
            outcomes=[[] for _ in range(4)]
            errors=[]
            def worker(rank):
                scheduler=BatchScheduler(trace,t0,windows[-1][1],rank,method,root,
                    mode=mode,margin=.05,factor=1,run_id='batch-test')
                try:
                    for n in range(count):
                        step=n*h+(h if h>1 else 0)
                        ok,opid=scheduler.gate('collective',step,1024,remaining_ops=count-n)
                        outcomes[rank].append((ok,time.time()-t0))
                        if not ok:break
                        time.sleep(.008*h)
                        scheduler.complete(opid,step,1024)
                        scheduler.finish_operation(opid,step)
                    self.assertFalse(scheduler.in_batch)
                except Exception as exc:
                    errors.append(repr(exc))
                finally:
                    scheduler.close()
            threads=[threading.Thread(target=worker,args=(rank,),daemon=True) for rank in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(windows[-1][1]+5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors,[])
            batches=[]
            for rank in range(4):
                with (root/f'e3_batch_rank{rank}.csv').open() as fp:
                    batches.append(list(csv.DictReader(fp)))
            counts=[[int(row['K']) for row in rs] for rs in batches]
            self.assertTrue(all(x==counts[0] for x in counts))
            self.assertTrue(all([ok for ok,_ in row]==[ok for ok,_ in outcomes[0]] for row in outcomes))
            for rs in batches:
                for row in rs:
                    if int(row['K']):
                        self.assertEqual(int(row['last_step'])-int(row['first_step']), (int(row['K'])-1)*h)
            return outcomes,counts[0]

    def test_ddp_batch_amortizes_grants(self):
        rows,counts=self.run_case('ddp','aware',[(0,3,1)],12)
        self.assertEqual(sum(counts),12)
        self.assertLess(len(counts),12)
        self.assertGreater(max(counts),1)
        self.assertTrue(all(len(row)==12 for row in rows))

    def test_repeated_contacts_and_stop(self):
        windows=[(0,.25,0),(.25,.6,1),(.6,1,0),(1,1.4,1),(1.4,2,0)]
        rows,counts=self.run_case('ddp','aware',windows,100)
        self.assertEqual(counts[-1],0)
        for row in rows:
            for ok,stamp in row:
                if ok:self.assertTrue(any(up and begin<=stamp<end for begin,end,up in windows),stamp)

    def test_fixed_diloco_h(self):
        rows,counts=self.run_case('diloco8','aware',[(0,4,1)],9)
        self.assertEqual(sum(counts),9)
        self.assertGreater(max(counts),1)

    def test_per_operation_ablation(self):
        _,counts=self.run_case('ddp','aware',[(0,3,1)],5,max_ops=1)
        self.assertEqual(counts,[1]*5)

    def test_no_future_contact(self):
        _,counts=self.run_case('ddp','aware',[(0,2,0)],5)
        self.assertEqual(counts,[0])


if __name__ == '__main__':
    unittest.main()
