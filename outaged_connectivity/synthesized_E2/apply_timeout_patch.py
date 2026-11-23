#!/usr/bin/env python3
"""Patch local filter_E2 scripts to configure NCCL and Flower round timeout.
Run in filter_E2: python3 /path/to/apply_timeout_patch.py
Refuses to patch files if the expected GitHub source fragments differ.
"""
from pathlib import Path
import sys, shutil, datetime
root = Path.cwd()
changes = {
 'train.py': [
  ("    p.add_argument('--log-every',type=int,default=20)\n", "    p.add_argument('--log-every',type=int,default=20)\n    p.add_argument('--comm-timeout',type=int,default=900,help='NCCL operation timeout in seconds')\n"),
  ("    a=p.parse_args()\n    os.environ['MASTER_ADDR']", "    a=p.parse_args()\n    if a.comm_timeout<=0:raise ValueError('comm-timeout must be positive')\n    os.environ['MASTER_ADDR']"),
  ("timeout=datetime.timedelta(seconds=180))", "timeout=datetime.timedelta(seconds=a.comm_timeout))")
 ],
 'train_flower.py': [
  ("    ap.add_argument('--seed',type=int,default=42)\n", "    ap.add_argument('--seed',type=int,default=42)\n    ap.add_argument('--round-timeout',type=float,default=900,help='Flower round timeout in seconds')\n"),
  ("    a=ap.parse_args()\n    while time.time()", "    a=ap.parse_args()\n    if a.round_timeout<=0:raise ValueError('round-timeout must be positive')\n    while time.time()"),
  ("config=fl.server.ServerConfig(num_rounds=10000),strategy=strategy,", "config=fl.server.ServerConfig(num_rounds=10000,round_timeout=a.round_timeout),strategy=strategy,")
 ],
 'runner.py': [
  ("    p.add_argument('--eval-every', type=int, default=100)\n", "    p.add_argument('--eval-every', type=int, default=100)\n    p.add_argument('--comm-timeout', type=int, default=int(os.getenv('E2_COMM_TIMEOUT', '900')))\n"),
  ("    a = p.parse_args()\n    if not 60", "    a = p.parse_args()\n    if a.comm_timeout <= 0:\n        raise ValueError('comm-timeout must be positive')\n    if not 60"),
  ("        dataset=a.dataset), indent=2))", "        dataset=a.dataset, comm_timeout_s=a.comm_timeout), indent=2))"),
  ("'--output', str(out), '--dataset', a.dataset]\n            launch(nodes[0]['host'], 'flower_server'", "'--output', str(out), '--dataset', a.dataset,\n                      '--round-timeout', str(a.comm_timeout)]\n            launch(nodes[0]['host'], 'flower_server'"),
  ("'--output', str(out), '--dataset', a.dataset]\n                cmd = ['env', 'CUDA_VISIBLE_DEVICES=0'", "'--output', str(out), '--dataset', a.dataset,\n                          '--round-timeout', str(a.comm_timeout)]\n                cmd = ['env', 'CUDA_VISIBLE_DEVICES=0'"),
  ("'--eval-every', str(a.eval_every), '--log-every', os.getenv('E2_LOG_EVERY', '20')]", "'--eval-every', str(a.eval_every), '--log-every', os.getenv('E2_LOG_EVERY', '20'),\n                       '--comm-timeout', str(a.comm_timeout)]")
 ]
}
prepared={}
for name, edits in changes.items():
 p=root/name
 if not p.exists():
  sys.exit(f'ERROR: {p} not found. Run from filter_E2 directory.')
 content=p.read_text()
 updated=content
 for old,new in edits:
  if old in updated:
   if updated.count(old)!=1:
    sys.exit(f'ERROR: {name}: expected unique source fragment: {old!r}')
   updated=updated.replace(old,new,1)
  elif new in updated:
   print(f'Already applied: {name}: {old[:45]!r}')
  else:
   sys.exit(f'ERROR: {name}: source changed; refusing partial patch at {old!r}')
 prepared[name]=updated
# Compilation checks happen before any original is modified.
import ast
for name, data in prepared.items():
 ast.parse(data,filename=name)
for name, data in prepared.items():
 p=root/name
 if data==p.read_text():
  continue
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
 backup=p.with_name(name+'.before_timeout_'+stamp)
 shutil.copy2(p,backup)
 p.write_text(data)
 print(f'Updated {p}, backup {backup}')
print('SUCCESS. Check: python3 -m py_compile train.py train_flower.py runner.py')
