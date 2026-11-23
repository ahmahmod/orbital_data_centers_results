# E2_FULL_METRICS_V1
#!/usr/bin/env python3
"""Four-GPU E2 coordinator with remote PID ownership and trustworthy exit status.

Works with existing run.sh, remote.py, train.py, train_flower.py and analyze.py.
Remote workers are launched in their own sessions, and are terminated by verified
PID/start-time after failure. Shaped tc agents perform their own qdisc cleanup.
No helper files or privileged remote daemons are required.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

from remote import ROOT, read_state, ssh, remote_argv, route_guard
from network_select import validate_shared_routes


def sq(s):
    return shlex.quote(str(s))


def remote_pid_wrapper(argv, marker, cwd):
    """Start a remote session leader and save its PID + /proc start ticks.

    setsid -w waits for the child exit; exec causes session leader to become
    the requested worker process. The marker is local to the remote node.
    """
    body = (
        'set -e; '
        'printf "%s %s\\n" "$$" "$(awk \'{print $22}\' /proc/$$/stat)" > ' + sq(marker) + '; '
        'exec ' + shlex.join(map(str, argv))
    )
    return 'cd ' + sq(cwd) + ' && exec setsid -w bash -c ' + sq(body)


# Runs on the target node. Marker PID is trusted only when it still denotes
# the same process, process group leader, and command belonging to this run.
REMOTE_STOP = r'''
import os, pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1]); unique = sys.argv[2]
if not marker.exists():
    print('No remote PID marker:', marker)
    sys.exit(0)
try:
    pid_s, ticks = marker.read_text().strip().split()
    pid = int(pid_s)
    stat = pathlib.Path(f'/proc/{pid}/stat')
    if not stat.exists():
        marker.unlink(missing_ok=True); sys.exit(0)
    # account for parentheses in /proc/PID/stat comm field
    rest = stat.read_text().rsplit(')', 1)[1].strip().split()
    actual_ticks = rest[19]  # proc stat field 22
    raw = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    is_worker = any(name in raw for name in ('train.py', 'train_flower.py', 'tc_agent.py'))
    if actual_ticks != ticks or unique not in raw or not is_worker or os.getpgid(pid) != pid:
        print('REFUSING: PID identity did not match', marker, pid, raw[:200], file=sys.stderr)
        sys.exit(7)
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not stat.exists():
            break
        time.sleep(0.2)
    if stat.exists():
        # Force-stop only the verified same process group after grace period.
        if stat.read_text().rsplit(')', 1)[1].strip().split()[19] == ticks and os.getpgid(pid) == pid:
            print('Escalating to SIGKILL:', pid, file=sys.stderr)
            os.killpg(pid, signal.SIGKILL)
    marker.unlink(missing_ok=True)
except ProcessLookupError:
    marker.unlink(missing_ok=True)
except Exception as e:
    print('Remote termination error:', repr(e), file=sys.stderr)
    sys.exit(8)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ddp', 'gossip8', 'gossip64', 'diloco8', 'diloco64', 'fl8', 'fl64'], required=True)
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--startup-lead', type=int, default=150)
    p.add_argument('--shutdown-grace', type=int, default=int(os.getenv('E2_SHUTDOWN_GRACE', '300')))
    p.add_argument('--apply', action='store_true')
    p.add_argument('--rate-scale', type=float, default=1)
    p.add_argument('--out', default='')
    p.add_argument('--python', default=os.getenv('E2_PYTHON', str(ROOT / '.venv/bin/python')))
    p.add_argument('--dataset', default='cifar100', choices=['cifar100', 'synthetic'])
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--trace', default=str(ROOT / 'traces/directed_trace_4workers.csv'))
    p.add_argument('--master-port', type=int, default=29527)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--flower-rounds', type=int, default=int(os.getenv('E2_FLOWER_ROUNDS', '500')))
    p.add_argument('--eval-interval-s',type=float,default=float(os.getenv('E2_EVAL_INTERVAL_S','300')))
    p.add_argument('--resource-interval-s',type=float,default=float(os.getenv('E2_RESOURCE_INTERVAL_S','2')))
    p.add_argument('--comm-timeout', type=int, default=int(os.getenv('E2_COMM_TIMEOUT', '900')))
    a = p.parse_args()
    if a.comm_timeout <= 0:
        raise ValueError('comm-timeout must be positive')
    if a.flower_rounds <= 0:
        raise ValueError('flower-rounds must be positive')
    if not 60 <= a.max_seconds <= 18000:
        raise RuntimeError('Duration must be between 60 and 18000 seconds')
    if a.shutdown_grace < 60:
        raise RuntimeError('Shutdown grace must be >= 60 seconds')
    if a.startup_lead < 60:
        raise RuntimeError('Startup lead must be >= 60 seconds')

    x = read_state()
    nodes = sorted(x['nodes'], key=lambda n: n['rank'])
    mode = x.get('network_mode', 'kavlan')
    if mode not in ('kavlan', 'selective_tc'):
        raise RuntimeError('Unsupported network mode')
    if len(nodes) != 4 or [n['rank'] for n in nodes] != list(range(4)):
        raise RuntimeError('Exactly four ranks 0..3 required')
    if len({n['iface'] for n in nodes}) != 1:
        raise RuntimeError('E2 requires a common experimental interface name')
    if mode == 'selective_tc':
        validate_shared_routes(nodes)
    if a.apply and os.getenv('E2_CONFIRM_SHAPING') != '1':
        raise RuntimeError('Set E2_CONFIRM_SHAPING=1 before changing network qdiscs')
    if a.apply and mode == 'selective_tc' and os.getenv('E2_CONFIRM_SHARED_TC') != '1':
        raise RuntimeError('Set E2_CONFIRM_SHARED_TC=1 before changing the management NIC')

    trace = pathlib.Path(a.trace).resolve()
    python = pathlib.Path(os.path.abspath(a.python))  # DO NOT resolve venv symlink!
    if not trace.is_file():
        raise RuntimeError(f'Contact trace missing: {trace}')
    if not python.is_file():
        raise RuntimeError(f'Virtual environment Python missing: {python}')
    dataset = ROOT / 'data'
    if a.dataset == 'cifar100' and not (dataset / 'cifar-100-python').is_dir():
        raise RuntimeError('Missing CIFAR-100 dataset; run download_cifar.py')

    out = (pathlib.Path(a.out).resolve() if a.out else
           ROOT / 'results' / ('E2_' + a.method + '_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    if out.exists():
        raise RuntimeError(f'Output directory already exists: {out}')

    # Fail before any tc changes if an old root qdisc is still present.
    if a.apply:
        for n in nodes:
            q = ssh(n['host'], 'tc qdisc show dev ' + sq(n['iface']))
            if any(line.startswith('qdisc htb 1: root') for line in q.splitlines()):
                raise RuntimeError(f"Stale HTB 1: root qdisc on {n['host']}:{n['iface']}; inspect cleanup.sh")

    print('Four allocated workers:', [(n['host'], n['ip'], n['iface']) for n in nodes], flush=True)
    for n in nodes:
        host = n['host']
        if mode == 'kavlan':
            route_guard(host, n['iface'])
        gpu = ssh(host, 'nvidia-smi --query-gpu=name --format=csv,noheader | head -1')
        if not gpu:
            raise RuntimeError(f'No NVIDIA GPU on {host}')
        # Real CUDA computation, not just is_available(). Catches P100/V100/Blackwell wheel mismatches.
        probe = (
            'import torch, torchvision' + (',flwr' if a.method.startswith('fl') else '') + '; '
            'assert torch.cuda.is_available(); '
            'x=torch.ones((32,32),device="cuda"); y=x@x; torch.cuda.synchronize(); '
            'assert y.sum().item()==32768.0; '
            'print(torch.__version__,torch.cuda.get_device_name(0))'
        )
        print(host, ssh(host, shlex.join([str(python), '-c', probe]), timeout=100), flush=True)
        for needed in [ROOT / 'train.py', ROOT / 'train_flower.py', ROOT / 'tc_agent.py', trace]:
            ssh(host, 'test -r ' + sq(needed))
        if a.dataset == 'cifar100':
            ssh(host, 'test -d ' + sq(dataset / 'cifar-100-python'))
        if mode == 'kavlan':
            for peer in nodes:
                if peer is not n:
                    route_guard(host, n['iface'], peer['ip'], n['ip'])

    out.mkdir(parents=True)
    (out / 'nodes.json').write_text(json.dumps(dict(
        nodes=nodes, oar_job_id=x['oar_job_id'], site=x['site'],
        vlan_id=x.get('vlan_id'), network_mode=mode), indent=2))
    start = time.time() + a.startup_lead
    (out / 'e2_run_config.json').write_text(json.dumps(dict(
        method=a.method, init_unix_s=start, duration_s=a.max_seconds, nodes=nodes,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        tc_applied=a.apply, network_mode=mode, rate_scale=a.rate_scale,
        dataset=a.dataset, comm_timeout_s=a.comm_timeout, shutdown_grace_s=a.shutdown_grace,
        training_profile=('E1_b_gossip_epoch_sync_v2' if a.method.startswith('gossip') else 'E1_aligned_v1'),
        validation_schedule=('every_epoch' if a.method.startswith('gossip') else 'wall_clock'),
        max_epochs=100, flower_rounds=a.flower_rounds if a.method.startswith('fl') else None,
        flower_steps_per_round=(8 if a.method=='fl8' else 64) if a.method.startswith('fl') else None,
        model='ResNet18_CIFAR_stem',
        outer_lr=float(os.getenv('E2_OUTER_LR', '0.3')),
        outer_momentum=float(os.getenv('E2_OUTER_MOMENTUM', '0.9')),
        seed=2026, batch_size=32, num_workers=a.num_workers, lr=0.1,
        lr_schedule='warmup5_cosine100', weight_decay=0.0005,
        amp_fp16=True, validation_source='train-holdout', val_size=5000,
        eval_interval_s=a.eval_interval_s,resource_interval_s=a.resource_interval_s), indent=2))

    run_id = uuid.uuid4().hex[:12]
    tasks = []
    logs = []
    stream_done = threading.Event()
    interrupted = False
    failures = []
    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f'Coordinator received signal {signum}; stopping remote workers...', flush=True)
    signal.signal(signal.SIGINT, on_interrupt)
    signal.signal(signal.SIGTERM, on_interrupt)

    def launch(host, role, argv, logpath):
        marker = f'/tmp/e2_{run_id}_{role}.pid'
        if mode == 'selective_tc' and a.apply:
            argv = ['env', 'E2_CONFIRM_SHARED_TC=1', *argv]
        command = remote_pid_wrapper(argv, marker, ROOT)
        fp = open(logpath, 'w', buffering=1)
        logs.append(fp)
        proc = subprocess.Popen(remote_argv(host, command), stdout=fp,
                                stderr=subprocess.STDOUT, start_new_session=True)
        task = dict(host=host, role=role, marker=marker, proc=proc, path=logpath)
        tasks.append(task)
        return proc

    def stream_rank0():
        path = out / 'train_rank0.log'
        while not path.exists() and not stream_done.is_set():
            time.sleep(0.1)
        if not path.exists():
            return
        with open(path) as fp:
            while True:
                line = fp.readline()
                if line:
                    print('[rank0] ' + line.rstrip(), flush=True)
                elif stream_done.is_set():
                    break
                else:
                    time.sleep(0.2)

    status = {}
    epochs_finished = False
    try:
        for n in nodes:
            cmd = [str(python), str(ROOT / 'tc_agent.py'), '--state', str(out / 'nodes.json'),
                   '--trace', str(trace), '--rank', str(n['rank']), '--start', str(start),
                   '--duration', str(a.max_seconds), '--out', str(out / f'replay_rank{n["rank"]}.csv'),
                   '--rate-scale', str(a.rate_scale)]
            if a.apply:
                cmd.append('--apply')
            launch(n['host'], f'tc{n["rank"]}', cmd, out / f'tc_rank{n["rank"]}.log')

        if a.method.startswith('fl'):
            flower = [str(python), str(ROOT / 'train_flower.py'), '--role', 'server',
                      '--server', nodes[0]['ip'] + ':8085', '--start', str(start),
                      '--duration', str(a.max_seconds), '--rounds', str(a.flower_rounds), '--steps-per-round',
                      '8' if a.method == 'fl8' else '64', '--data-root', str(dataset),
                      '--output', str(out), '--dataset', a.dataset,
                      '--round-timeout', str(a.comm_timeout),
                      '--eval-interval-s',str(a.eval_interval_s),'--resource-interval-s',str(a.resource_interval_s)]
            launch(nodes[0]['host'], 'flower_server', flower, out / 'flower_server.log')

        for n in nodes:
            rank = n['rank']
            if a.method.startswith('fl'):
                flower = [str(python), str(ROOT / 'train_flower.py'), '--role', 'client',
                          '--rank', str(rank), '--server', nodes[0]['ip'] + ':8085',
                          '--start', str(start), '--duration', str(a.max_seconds),
                          '--rounds', str(a.flower_rounds), '--steps-per-round', '8' if a.method == 'fl8' else '64',
                          '--data-root', str(dataset), '--output', str(out), '--dataset', a.dataset,
                          '--round-timeout', str(a.comm_timeout),
                          '--eval-interval-s',str(a.eval_interval_s),'--resource-interval-s',str(a.resource_interval_s)]
                cmd = ['env', 'E2_NET_IFACE='+n['iface'], 'CUDA_VISIBLE_DEVICES=0', 'OMP_NUM_THREADS=2',
                       'PYTHONUNBUFFERED=1', *flower]
            else:
                env = dict(MASTER_ADDR=nodes[0]['ip'], MASTER_PORT=str(a.master_port),
                           NCCL_SOCKET_IFNAME=n['iface'], GLOO_SOCKET_IFNAME=n['iface'],
                           NCCL_IB_DISABLE='1', CUDA_VISIBLE_DEVICES='0',
                           OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                cmd = ['env', *[f'{k}={v}' for k, v in env.items()], str(python),
                       str(ROOT / 'train.py'), '--method', a.method, '--rank', str(rank),
                       '--world', '4', '--master', nodes[0]['ip'], '--port', str(a.master_port),
                       '--start', str(start), '--duration', str(a.max_seconds),
                       '--data-root', str(dataset), '--output', str(out),
                       '--num-workers', str(a.num_workers), '--dataset', a.dataset,
                       '--eval-every', str(1 if a.method.startswith('gossip') else a.eval_every), '--log-every', os.getenv('E2_LOG_EVERY', '20'),
                       '--comm-timeout', str(a.comm_timeout),
                       '--outer-lr', os.getenv('E2_OUTER_LR', '0.3'),
                       '--outer-momentum', os.getenv('E2_OUTER_MOMENTUM', '0.9'),
                       '--eval-interval-s',str(a.eval_interval_s),'--resource-interval-s',str(a.resource_interval_s)]
            launch(n['host'], f'train{rank}', cmd, out / f'train_rank{rank}.log')

        threading.Thread(target=stream_rank0, daemon=True).start()
        print(f'Launched {len(tasks)} remote processes, start={start:.3f}, run={run_id}', flush=True)
        deadline = start + a.max_seconds + a.shutdown_grace
        print(f'Training ends at {start + a.max_seconds:.3f}; shutdown grace {a.shutdown_grace}s', flush=True)
        while time.time() < deadline and not interrupted:
            status = {str(t['path'].name): t['proc'].poll() for t in tasks}
            if any(v is not None and v != 0 for v in status.values()):
                failures.append('One or more remote processes exited nonzero')
                break
            # Replay agents intentionally run for the full scheduled duration.
            # For non-Flower methods, do not wait for them once every trainer
            # has completed successfully (e.g., after all 100 epochs).
            trainers = [t for t in tasks if t['role'].startswith('train')]
            if a.method.startswith('fl'):
                # Flower clients terminate after the fixed-round server shuts
                # down. Require the server's complete summary as well as all
                # five process exit codes before considering this run finished.
                server = next(t for t in tasks if t['role'] == 'flower_server')
                if server['proc'].poll() == 0 and len(trainers) == len(nodes) and all(
                    t['proc'].poll() == 0 for t in trainers
                ):
                    summary_path = out / 'summary.json'
                    try:
                        summary = json.loads(summary_path.read_text())
                        complete = (summary.get('status') == 'ok'
                            and int(summary.get('completed_rounds', -1)) == a.flower_rounds
                            and int(summary.get('evaluated_rounds', -1)) == a.flower_rounds)
                    except (OSError, ValueError, TypeError):
                        complete = False
                    if not complete:
                        failures.append('Flower workers exited but fixed-round summary is missing or incomplete')
                    else:
                        epochs_finished = True
                        print(f'FLOWER COMPLETE: {a.flower_rounds} rounds and global evaluations finished; '
                              'stopping contact replay and finalizing.', flush=True)
                    break
            elif len(trainers) == len(nodes) and all(t['proc'].poll() == 0 for t in trainers):
                epochs_finished = True
                print('TRAINING COMPLETE: all four workers exited successfully; '
                      'stopping contact replay and finalizing this method.', flush=True)
                break
            if all(v is not None for v in status.values()):
                break
            time.sleep(1)
        else:
            if not interrupted and any(t['proc'].poll() is None for t in tasks):
                pending = [t['role'] + '@' + t['host'] for t in tasks if t['proc'].poll() is None]
                failures.append(f'Experiment exceeded wall-clock deadline; pending: {pending}')
    except Exception as exc:
        failures.append(repr(exc))
        print('Coordinator error:', repr(exc), file=sys.stderr, flush=True)
    finally:
        if interrupted:
            failures.append('Coordinator interrupted')
        # Terminate active remote processes by verified PID, not just local SSH.
        # Stop training before tc, so tc has time to remove its own qdisc.
        print('FINALIZATION: stopping remaining training workers before tc agents', flush=True)
        for role_prefix in ('train', 'flower_server', 'tc'):
            for t in tasks:
                if not t['role'].startswith(role_prefix):
                    continue
                if t['proc'].poll() is None:
                    command = 'python3 -c ' + sq(REMOTE_STOP) + ' ' + sq(t['marker']) + ' ' + sq(str(out))
                    try:
                        result = ssh(t['host'], command, timeout=25)
                        if result:
                            print(f'Remote stop {t["host"]}/{t["role"]}: {result}', flush=True)
                    except Exception as exc:
                        failures.append(f'Remote stop failed {t["host"]}/{t["role"]}: {exc}')
        # Give tc agents a short opportunity to finish their own cleanup before
        # closing any still-running SSH client. Never delete qdiscs blindly.
        for t in tasks:
            if t['role'].startswith('tc') and t['proc'].poll() is None:
                try:
                    t['proc'].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print('TC agent still pending after cleanup grace:', t['host'], flush=True)
        for t in tasks:
            if t['proc'].poll() is None:
                try:
                    t['proc'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(t['proc'].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        t['proc'].wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(t['proc'].pid, signal.SIGKILL)
                        t['proc'].wait()
        for fp in logs:
            fp.close()
        stream_done.set()
        status = {str(t['path'].name): t['proc'].returncode for t in tasks}
        bad = {name: code for name, code in status.items() if code != 0}
        if epochs_finished:
            # tc_agent handles SIGTERM and removes its *own* qdisc in finally.
            # A remote SSH launcher may return a signal exit code even when
            # the agent confirms its cleanup; accept only verified cleanup.
            for t in tasks:
                if not t['role'].startswith('tc'):
                    continue
                name = t['path'].name
                if name not in bad:
                    continue
                content = t['path'].read_text(errors='replace')
                confirmed = ('E2 tc cleanup PASS:' in content if a.apply
                             else 'E2 dry run:' in content)
                if confirmed:
                    print(f'Replay agent {name} confirmed cleanup after early stop '
                          f'(launcher exit {bad[name]}).', flush=True)
                    bad.pop(name)
        if bad:
            failures.append(f'Nonzero process exits: {bad}')
        # Check for leftover E2-style qdisc and local tc-agent ownership marker.
        if a.apply:
            for n in nodes:
                try:
                    q = ssh(n['host'], 'tc qdisc show dev ' + sq(n['iface']))
                    if any(line.startswith('qdisc htb 1: root') for line in q.splitlines()):
                        failures.append(f"Residual HTB qdisc on {n['host']}:{n['iface']}")
                except Exception as exc:
                    failures.append(f"Cannot verify qdisc on {n['host']}: {exc}")
        (out / 'execution_status.json').write_text(json.dumps(dict(
            return_codes=status, failures=failures, shutdown_grace_s=a.shutdown_grace, wall_end=time.time(),
            run_id=run_id, success=(not failures),
            training_completed_early=epochs_finished,
            completion_reason=('flower_rounds_completed' if epochs_finished and a.method.startswith('fl') else
                               'trainers_finished' if epochs_finished else
                               'interrupted' if interrupted else 'duration_or_failure')), indent=2))
        print('Log/output folder:', out, flush=True)
        subprocess.run([sys.executable, str(ROOT / 'analyze.py'), str(out)], check=False)
        subprocess.run([sys.executable, str(ROOT / 'plot_e2_full.py'), str(out)], check=False)
    if failures:
        print('E2 FAILED:', *failures, sep='\n - ', file=sys.stderr)
        return 1
    print('E2 SUCCESS:', out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
