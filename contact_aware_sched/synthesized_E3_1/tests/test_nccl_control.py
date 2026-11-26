"""CPU-only tests of the real filesystem grant protocol; no GPUs required."""
import csv
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e3_control import ContactScheduler


class GrantTest(unittest.TestCase):
    def exercise(self, mode, windows, count, duration=8):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / 'trace.csv'
            with trace.open('w') as fp:
                writer = csv.writer(fp)
                writer.writerow(['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms'])
                for i in range(1,5):
                    for j in range(1,5):
                        if i == j:
                            continue
                        for start,end,up in windows:
                            writer.writerow([f'SAT-DEMO-{i:03d}',f'SAT-DEMO-{j:03d}',start,end,up,1000000000,1])
            start = time.time() + .15
            outcomes = [[] for _ in range(4)]
            errors = []
            def worker(rank):
                scheduler = ContactScheduler(trace,start,duration,rank,'ddp',root,
                    mode=mode,margin=.02,min_transfer=.02,factor=1,run_id='test')
                scheduler.open_guard=.02
                scheduler.launch_guard=.02
                scheduler.timeout=3
                try:
                    for step in range(count):
                        ok,opid=scheduler.gate('collective',step,1024)
                        outcomes[rank].append((ok,time.time()-start))
                        if not ok:
                            break
                        scheduler.complete(opid,step,1024)
                        time.sleep(.18)
                except Exception as exc:
                    errors.append(repr(exc))
                finally:
                    scheduler.close()
            threads=[threading.Thread(target=worker,args=(rank,),daemon=True) for rank in range(4)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(duration+5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors,[])
            self.assertTrue(all([ok for ok,_ in row]==[ok for ok,_ in outcomes[0]] for row in outcomes))
            return outcomes

    def test_repeated_contacts(self):
        windows=[(0,.2,0),(.2,.5,1),(.5,1,0),(1,1.35,1),(1.35,2,0),(2,8,1)]
        rows=self.exercise('aware',windows,6)
        for row in rows:
            self.assertEqual(len(row),6)
            for ok,stamp in row:
                self.assertTrue(ok)
                self.assertTrue(any(up and begin<=stamp<end for begin,end,up in windows),stamp)

    def test_no_future_contact(self):
        rows=self.exercise('aware',[(0,8,0)],1)
        self.assertTrue(all(row[0][0] is False for row in rows))

    def test_reactive_ignores_trace(self):
        rows=self.exercise('off',[(0,8,0)],3)
        self.assertTrue(all(len(row)==3 and all(ok for ok,_ in row) for row in rows))


if __name__ == '__main__':
    unittest.main()
