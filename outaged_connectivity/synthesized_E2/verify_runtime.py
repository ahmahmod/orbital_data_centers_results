#!/usr/bin/env python3
"""Non-destructive GPU+NFS checks on four deployed nodes before E2 training."""
import sys
from remote import hosts,ssh,ROOT

def main():
    failures=[]
    for h in hosts():
        print('CHECK',h,flush=True)
        for label,cmd in [
            ('NVIDIA', 'nvidia-smi -L'),
            ('shared package','test -r '+str(ROOT/'train.py')+' && echo available'),
            ('Python','python3 --version'),
        ]:
            try:print(label,ssh(h,cmd),flush=True)
            except Exception as e:failures.append(f'{h} {label}: {e}')
    if failures:
        for failure in failures:print('FAILED',failure,file=sys.stderr)
        return 1
    print('PASS: GPU driver and E2 directory accessible from all four deployed nodes')
    return 0
if __name__=='__main__':sys.exit(main())
