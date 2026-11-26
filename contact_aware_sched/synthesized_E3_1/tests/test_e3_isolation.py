#!/usr/bin/env python3
"""Offline architecture checks for technique-local E3_1 entrypoints."""
import ast,pathlib,unittest,sys
BASE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))

class TechniqueIsolationTests(unittest.TestCase):
    def test_each_technique_has_trainer_and_own_policy(self):
        for family in ('ddp','diloco','gossip','flower'):
            with self.subTest(family=family):
                self.assertTrue((BASE/family/'train.py').is_file())
                self.assertTrue((BASE/family/'schedule.py').is_file())
                ast.parse((BASE/family/'train.py').read_text())
                ast.parse((BASE/family/'schedule.py').read_text())

    def test_runner_selects_technique_local_trainers(self):
        code=(BASE/'runner.py').read_text()
        self.assertIn("trainer_path = ROOT / technique / 'train.py'",code)
        self.assertIn("schedule_path = ROOT / technique / 'schedule.py'",code)
        self.assertNotIn("str(ROOT / 'train.py')",code)
        self.assertNotIn("str(ROOT / 'train_flower.py')",code)

    def test_gossip_epoch_validation_has_no_collectives(self):
        tree=ast.parse((BASE/'gossip'/'train.py').read_text())
        nested=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef)
                and x.name=='validate_at_epoch']
        self.assertEqual(len(nested),1)
        src=ast.get_source_segment((BASE/'gossip'/'train.py').read_text(),nested[0])
        self.assertNotIn('dist.barrier(',src)
        self.assertNotIn('dist.broadcast(',src)
        self.assertNotIn("scheduler.gate('collective'",src)
        self.assertIn('local_validation_complete',src)

    def test_schedule_rejects_global_gossip_collectives(self):
        from importlib import util
        p=BASE/'gossip'/'schedule.py'
        spec=util.spec_from_file_location('gossip_policy_test',p)
        module=util.module_from_spec(spec);spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError,'per-peer'):
            module.ContactScheduler.gate(object(),'collective',0,4)

if __name__=='__main__':unittest.main()
