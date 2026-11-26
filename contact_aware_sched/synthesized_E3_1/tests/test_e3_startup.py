#!/usr/bin/env python3
"""Offline tests for DDP readiness barrier and trace-clock safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from e3_startup import (
    startup_dir,ready_status,publish_ready,publish_start,await_start,
    publish_failure,failure_status,
)

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.run_id='unit-run-1'

    def test_requires_all_four_current_run_workers(self):
        for rank in range(3):
            publish_ready(self.out,rank,self.run_id)
        self.assertEqual(sorted(ready_status(self.out,self.run_id)),[0,1,2])
        with self.assertRaisesRegex(RuntimeError,'all four'):
            publish_start(self.out,self.run_id,123456.)
        # A stale record from a different run does not count as readiness.
        publish_ready(self.out,3,'other-run')
        with self.assertRaisesRegex(RuntimeError,'all four'):
            publish_start(self.out,self.run_id,123456.)
        publish_ready(self.out,3,self.run_id)
        publish_start(self.out,self.run_id,123456.)
        self.assertEqual(await_start(self.out,self.run_id,timeout_s=1,poll_s=.01),123456.)

    def test_rejects_wrong_start_token(self):
        for rank in range(4):publish_ready(self.out,rank,self.run_id)
        publish_start(self.out,self.run_id,123456.)
        with self.assertRaisesRegex(TimeoutError,'did not publish'):
            await_start(self.out,'wrong-run',timeout_s=.04,poll_s=.01)

    def test_publishes_rank_failure(self):
        publish_failure(self.out,2,self.run_id,'fatal error')
        self.assertEqual(failure_status(self.out,self.run_id)[2]['detail'],'fatal error')
        self.assertEqual(failure_status(self.out,'unrelated'),{})

    def test_coordinator_orders_ddp_workers_before_tc(self):
        code=(BASE/'runner.py').read_text()
        self.assertLess(code.index('for n in nodes:\n            rank = n[\'rank\']'),
                        code.index("if a.method == 'ddp':\n            # No qdisc modification"))
        self.assertIn("if a.method != 'ddp':\n            launch_tc_agents()",code)
        self.assertIn('publish_start(out,run_id,start)\n            print(',code)
        self.assertIn("print('DDP AND ALL TC AGENTS READY;",code)
        self.assertLess(code.index("launch_tc_agents()\n            # TC agents"),code.index('publish_start(out,run_id,start)'))

if __name__=='__main__':unittest.main()
