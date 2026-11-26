"""CPU tests for pure reactive path and coordinated termination."""
import csv
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_reactive import ReactiveScheduler
from e3_batch_control import BatchScheduler


class ReactiveTest(unittest.TestCase):
    def exercise(self, method, duration, count, interval, expire_after=None):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {
                'E3_REACTIVE_STOP_CHECK_OPS':str(interval),
                'E3_REACTIVE_STOP_TIMEOUT_S':'0'}):
            root=Path(tmp)
            trace=root/'trace.csv'
            with trace.open('w') as fp:
                w=csv.writer(fp)
                w.writerow(['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
                for i in range(1,5):
                    for j in range(1,5):
                        if i!=j:
                            w.writerow([f'SAT-DEMO-{i:03d}',f'SAT-DEMO-{j:03d}',0,duration,0,0,1])
            t0=time.time()
            h=int(''.join(c for c in method if c.isdigit()) or '1')
            scheds=[ReactiveScheduler(trace,t0,duration,rank,method,root,
                        mode='off',run_id='test',comm_timeout=1) for rank in range(4)]
            # Deliberately make any contact prediction fail: OFF must not call it.
            for s in scheds:
                s.windows=lambda *_: self.fail('Reactive path consulted contact windows')
            results=[[] for _ in scheds]
            errors=[]
            def worker(rank):
                s=scheds[rank]
                try:
                    for n in range(count):
                        step=n*h+(h if h>1 else 0)
                        if rank==0 and expire_after is not None and n==expire_after:
                            s.end=time.time()-1
                        ok,op=s.gate('collective',step,1024,remaining_ops=count-n)
                        results[rank].append(ok)
                        if not ok:break
                        s.complete(op,step,1024)
                        s.finish_operation(op,step)
                        if rank==1:time.sleep(.003)
                except Exception as exc:
                    errors.append(repr(exc))
                finally:
                    s.close()
            threads=[threading.Thread(target=worker,args=(rank,),daemon=True) for rank in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors,[])
            self.assertTrue(all(x==results[0] for x in results))
            self.assertFalse((root/'.e3_control'/'nccl').exists())
            self.assertFalse(list(root.glob('e3_batch_rank*.csv')))
            checkpoints=list((root/'.e3_control'/'reactive_stop').glob('*.json'))
            return results[0],len(checkpoints)

    def test_ddp_no_grants_even_if_trace_all_down(self):
        result,checks=self.exercise('ddp',30,12,4)
        self.assertEqual(result,[True]*12)
        self.assertEqual(checks,3)

    def test_diloco_fixed_H(self):
        for method in ('diloco8','diloco64'):
            with self.subTest(method=method):
                result,checks=self.exercise(method,30,9,4)
                self.assertEqual(result,[True]*9)
                self.assertEqual(checks,3)

    def test_same_stop_operation_despite_rank_skew(self):
        result,checks=self.exercise('ddp',30,12,4,expire_after=2)
        self.assertEqual(result,[True]*4+[False])
        self.assertEqual(checks,2)


if __name__=='__main__':
    unittest.main()
