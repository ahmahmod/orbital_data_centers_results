#!/usr/bin/env python3
"""E2 directed-contact replay agent.

No --apply: record the trace only; never change qdiscs.
--apply: exclusive per-interface ownership, shaped peer egress, best-effort
cleanup on SIGINT/SIGTERM/normal completion. SIGKILL/power loss still requires
manual, verified cleanup. Shared-interface shaping remains operationally risky.
"""
import argparse
import csv
import fcntl
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from e3_broker import client_from_env


def command(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if p.returncode:
        raise RuntimeError(f"{' '.join(map(str, argv))}: {p.returncode}: {p.stderr.strip()} {p.stdout.strip()}")
    return p.stdout.strip()


def tc(*args):
    prefix = []
    if os.geteuid() != 0:
        if shutil.which('sudo-g5k'):
            prefix = ['sudo-g5k']
        elif shutil.which('sudo'):
            prefix = ['sudo', '-n']
        else:
            raise RuntimeError('Need root or passwordless sudo for tc')
    return command(prefix + ['tc', *map(str, args)])


def root_htb(iface):
    return bool(re.search(r'^qdisc htb 1: root\b', tc('qdisc', 'show', 'dev', iface), re.M))


def validate_iface(iface, address, peers, mode):
    if iface in ('lo', 'br0') or not re.fullmatch(r'[A-Za-z0-9_.:-]+', iface):
        raise RuntimeError(f'Unsafe interface name: {iface}')
    default = command(['ip', '-4', 'route', 'show', 'default'])
    is_management = bool(re.search(r'\bdev\s+' + re.escape(iface) + r'\b', default))
    if mode == 'kavlan' and is_management:
        raise RuntimeError(f'Isolated interface {iface} carries default management route')
    if mode == 'selective_tc' and not is_management:
        raise RuntimeError(f'Shared interface {iface} is not the management route')
    address_list = command(['ip', '-4', '-o', 'addr', 'show', 'dev', iface])
    if not re.search(r'\binet\s+' + re.escape(address) + r'/', address_list):
        raise RuntimeError(f'Interface {iface} does not have IP {address}')
    for peer in peers:
        route = command(['ip', '-4', 'route', 'get', peer])
        if not re.search(r'\bdev\s+' + re.escape(iface) + r'\b', route):
            raise RuntimeError(f'Peer {peer} not routed via {iface}: {route}')


def load_trace(path, source, peers, duration):
    plans = {p['sat']: [] for p in peers}
    with open(path, newline='') as fp:
        for row in csv.DictReader(fp):
            if row['src'] == source and row['dst'] in plans:
                plans[row['dst']].append((
                    float(row['t_start_s']), float(row['t_end_s']),
                    int(row['available']), float(row['rate_bps']),
                    float(row['prop_delay_ms'])
                ))
    for sat, sequence in plans.items():
        sequence.sort()
        if not sequence or abs(sequence[0][0]) > 1e-6 or sequence[-1][1] < duration - 1e-6:
            raise RuntimeError(f'Incomplete contact trace for {sat} / duration {duration}')
        if any(abs(prev[1] - cur[0]) > 1e-6 for prev, cur in zip(sequence, sequence[1:])):
            raise RuntimeError(f'Discontinuous trace for {sat}')
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True)
    p.add_argument('--trace', required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--duration', type=float, default=3600)
    p.add_argument('--out', required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--rate-scale', type=float, default=1.0)
    a = p.parse_args()
    if a.rate_scale <= 0 or a.duration <= 0:
        raise ValueError('Rate scale and duration must be positive')

    state = json.loads(pathlib.Path(a.state).read_text())
    nodes = sorted(state['nodes'], key=lambda n: n['rank'])
    me = next(n for n in nodes if n['rank'] == a.rank)
    peers = [n for n in nodes if n['rank'] != a.rank]
    mode = state.get('network_mode', 'kavlan')
    if mode not in ('kavlan', 'selective_tc'):
        raise RuntimeError(f'Unsupported network mode {mode}')
    if mode == 'selective_tc' and a.apply and os.getenv('E2_CONFIRM_SHARED_TC') != '1':
        raise RuntimeError('Shared management NIC requires E2_CONFIRM_SHARED_TC=1')
    iface = me['iface']
    validate_iface(iface, me['ip'], [n['ip'] for n in peers], mode)
    plans = load_trace(a.trace, me['sat'], peers, a.duration)

    events = []
    for i, peer in enumerate(peers, 1):
        for t0, t1, up, rate, delay in plans[peer['sat']]:
            if t0 < a.duration:
                events.append((t0, i, peer, t1, up, rate, delay))
    events.sort(key=lambda x: (x[0], x[1]))

    active = True
    def stop(_sig, _frame):
        nonlocal active
        active = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    lockfile = None
    installed = False
    marker_written = False
    marker_path = pathlib.Path('/run') / ('e2_tc_' + iface + '.json')
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = out.parent.name
    owner = {'run_id': run_id, 'pid': os.getpid(), 'interface': iface,
             'mode': mode, 'state': str(pathlib.Path(a.state).resolve()), 'created': time.time()}
    fp = None
    broker=client_from_env()
    cleanup_failed = False

    try:
        if a.apply:
            # Local flock prevents another cooperative E2 agent from installing
            # concurrent rules on the same physical interface.
            lock_path = pathlib.Path('/run/lock') / ('e2_tc_' + iface + '.lock')
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lockfile = open(lock_path, 'a+')
            try:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Another E2 tc agent owns {iface} (lock {lock_path})')
            if marker_path.exists():
                raise RuntimeError(f'Existing E2 ownership marker {marker_path}; inspect/clean manually')
            existing = tc('qdisc', 'show', 'dev', iface)
            if root_htb(iface):
                raise RuntimeError(f'Existing HTB 1: qdisc on {iface}; inspect and run cleanup.sh')
            # Refuse unknown root qdiscs rather than destroying unexpected state.
            root_lines = [line for line in existing.splitlines() if re.search(r'\broot\b', line)]
            permitted = ('mq ', 'fq_codel ', 'pfifo_fast ', 'noqueue ', 'fq ', 'cake ')
            if any(not any(line.startswith('qdisc ' + x) for x in permitted) for line in root_lines):
                raise RuntimeError(f'Unexpected root qdisc on {iface}: {root_lines}')

            # The lock is held throughout installation, use, and cleanup.
            tc('qdisc', 'replace', 'dev', iface, 'root', 'handle', '1:', 'htb', 'default', '999')
            installed = True
            marker_path.write_text(json.dumps(owner, indent=2))
            marker_written = True
            tc('class', 'add', 'dev', iface, 'parent', '1:', 'classid', '1:999',
               'htb', 'rate', '10gbit')
            if mode == 'selective_tc':
                tc('filter', 'add', 'dev', iface, 'protocol', 'ip', 'parent', '1:',
                   'prio', '1', 'flower', 'ip_proto', 'tcp', 'src_port', '22', 'classid', '1:999')
                tc('filter', 'add', 'dev', iface, 'protocol', 'ip', 'parent', '1:',
                   'prio', '2', 'flower', 'ip_proto', 'tcp', 'dst_port', '22', 'classid', '1:999')
            for i, peer in enumerate(peers, 1):
                tc('class', 'add', 'dev', iface, 'parent', '1:', 'classid', f'1:{i}',
                   'htb', 'rate', '10gbit', 'ceil', '10gbit')
                tc('qdisc', 'add', 'dev', iface, 'parent', f'1:{i}',
                   'handle', f'{i * 10}:', 'netem', 'delay', '1ms')
                tc('filter', 'add', 'dev', iface, 'protocol', 'ip', 'parent', '1:',
                   'prio', str(i + 10), 'u32', 'match', 'ip', 'dst', peer['ip'] + '/32',
                   'flowid', f'1:{i}')
            print(f'E2 tc installed: iface={iface}, mode={mode}, run={run_id}', flush=True)
        else:
            print(f'E2 dry run: rank={a.rank}, iface={iface}, no qdisc modifications', flush=True)

        fp = out.open('w', newline='', buffering=1)
        writer = csv.writer(fp)
        writer.writerow(['rank', 'src', 'dst', 't_start_s', 't_end_s', 'available',
                         'rate_bps', 'prop_delay_ms', 'applied_unix_s'])
        for t0, i, peer, t1, up, rate, delay in events:
            if not active:
                break
            while active and time.time() < a.start + t0:
                time.sleep(min(0.1, max(0.01, a.start + t0 - time.time())))
            if not active:
                break
            if a.apply:
                mbps = max(rate * a.rate_scale / 1e6, 0.001) if up else 0.001
                tc('class', 'replace', 'dev', iface, 'parent', '1:', 'classid', f'1:{i}',
                   'htb', 'rate', f'{mbps:.5f}mbit', 'ceil', f'{mbps:.5f}mbit')
                tc('qdisc', 'replace', 'dev', iface, 'parent', f'1:{i}',
                   'handle', f'{i * 10}:', 'netem', 'delay', f'{max(delay, 0.01):.4f}ms',
                   'loss', '0%' if up else '100%')
            applied=time.time()
            if broker:
                broker.write(f"replay/rank{a.rank}/{peer['sat']}",dict(
                    run_id=os.environ['E3_RUN_ID'],t_start_s=t0,t_end_s=t1,
                    available=up,rate_bps=int(rate*a.rate_scale) if up else 0,
                    applied_unix_s=applied))
            writer.writerow([a.rank, me['sat'], peer['sat'], t0, t1, up,
                             int(rate * a.rate_scale) if up else 0, delay, applied])
        # Hold the final scheduled link state through the end of the window.
        while active and time.time() < a.start + a.duration:
            time.sleep(min(0.1, max(0.01, a.start + a.duration - time.time())))
        if not active:
            print('E2 tc agent received stop signal; cleaning up', flush=True)
    finally:
        if fp:
            fp.close()
        if installed:
            try:
                # Only clean when still holding the lock and metadata has not been replaced.
                if marker_written and json.loads(marker_path.read_text()) != owner:
                    raise RuntimeError(f'Ownership marker changed: {marker_path}')
                if not root_htb(iface):
                    raise RuntimeError('Owned root HTB qdisc missing or changed; refusing blind deletion')
                tc('qdisc', 'del', 'dev', iface, 'root')
                print(f'E2 tc cleanup PASS: {iface}', flush=True)
                if marker_written:
                    marker_path.unlink(missing_ok=True)
            except Exception as exc:
                cleanup_failed = True
                print(f'CRITICAL: E2 tc cleanup failed on {iface}: {exc}', file=sys.stderr, flush=True)
        if lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
            lockfile.close()
        if broker:broker.close()
        if cleanup_failed and sys.exc_info()[0] is None:
            raise RuntimeError(f'Cleanup failed for {iface}; inspect tc state before next run')


if __name__ == '__main__':
    main()
