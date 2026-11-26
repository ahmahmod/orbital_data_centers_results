#!/usr/bin/env bash
# Sequential, resumable E3_1 campaign. Run on frontend inside prepared OAR job.
set -Eeuo pipefail
cd "$(dirname "$(realpath "$0")")"
: "${MAX_SECONDS:=10800}"
: "${DATASET:=cifar100}"
: "${AUTO_RETRY_FAILED:=0}"
: "${AUTO_START_BUFFER_S:=2400}"
: "${AUTO_REPETITION:=R01}"
: "${AUTO_DRY_RUN:=0}"
: "${E2_CONFIRM_SHARED_TC:=0}"
export MAX_SECONDS DATASET AUTO_RETRY_FAILED AUTO_REPETITION
export TC_APPLY=1 E2_CONFIRM_SHAPING=1 E2_CONFIRM_SHARED_TC
export E2_COMM_TIMEOUT=900 E2_LOG_EVERY=20 E2_EVAL_INTERVAL_S=300 E2_RESOURCE_INTERVAL_S=2 E2_SHUTDOWN_GRACE=300
export E2_OUTER_LR=0.3 E2_OUTER_MOMENTUM=0.9
export E3_NCCL_TOPOLOGY=ring_auto E3_GRANT_MAX_OPS=32 E3_MARGIN_S=15 E3_ESTIMATE_FACTOR=2
export E3_MIN_TRANSFER_S=.5 E3_LOCAL_STEP_BUDGET_S=.25 E3_OBSERVED_FACTOR=1.5
export E3_OBSERVED_WINDOW=32 E3_OBSERVED_MAX_AGE_S=120 E3_OBSERVED_QUANTILE=.9
export E3_CONTACT_OPEN_GUARD_S=.25 E3_LAUNCH_GUARD_S=.25 E3_CONTROL_POLL_S=.05 E3_RENDEZVOUS_TIMEOUT=180
export E3_REACTIVE_STOP_CHECK_OPS=32 E3_REACTIVE_STOP_TIMEOUT_S=0
export E3_RING_PORT=29528 E3_RING_FRAME_BYTES=1048576 E3_RING_MARGIN_S=.5 E3_RING_FACTOR=2
export E3_RING_OPEN_GUARD_S=.25 E3_RING_MIN_TRANSFER_S=.02 E3_RING_CONNECT_TIMEOUT_S=180
export E3_RING_DRAIN_GUARD_S=2 E3_RING_DRAIN_TIMEOUT_S=240.0
export STARTUP_LEAD=150 E3_STARTUP_READY_TIMEOUT=1800
export E2_REMOTE_MODE=deploy
[[ "$MAX_SECONDS" =~ ^[0-9]+$ && "$AUTO_START_BUFFER_S" =~ ^[0-9]+$ ]] || { echo 'Invalid duration/buffer'; exit 2; }
[[ "$AUTO_REPETITION" =~ ^R[0-9]+$ ]] || { echo 'AUTO_REPETITION must be R01, R02, ...'; exit 2; }
[[ "$AUTO_RETRY_FAILED" =~ ^[01]$ && "$AUTO_DRY_RUN" =~ ^[01]$ ]] || exit 2
TRACE="$PWD/traces/directed_trace_4workers.csv"
[[ -f "$TRACE" ]] || exit 2
export AUTO_TRACE="$TRACE"
# Check trace coverage before spending a reservation.
python3 - <<'PY'
import csv,os
from collections import defaultdict
edges=defaultdict(float)
with open(os.environ['AUTO_TRACE']) as f:
    for r in csv.DictReader(f):
        k=r['src'],r['dst']
        edges[k]=max(edges[k],float(r['t_end_s']))
assert len(edges)==12 and min(edges.values())>=int(os.environ['MAX_SECONDS']), 'Trace is shorter than requested duration'
PY
mkdir -p auto_logs
# Lock is for this campaign checkout; do not launch manual experiments concurrently.
exec 9>auto_logs/campaign.lock
flock -n 9 || { echo 'Another auto.sh is active'; exit 2; }
LOG="auto_logs/campaign_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "Campaign log: $LOG"
echo "Maximum training budget: 12 x $MAX_SECONDS seconds; reservation/setup/cleanup extra."
echo "Repetition=$AUTO_REPETITION; retry failed=$AUTO_RETRY_FAILED; dry run=$AUTO_DRY_RUN"

if [[ "$AUTO_DRY_RUN" != 1 ]]; then
    : "${OAR_JOB_ID:?Run inside an active prepared OAR reservation}"
    python3 - <<'PY'
import os
from remote import read_state
s=read_state()
if s.get('network_mode')=='selective_tc' and os.environ['E2_CONFIRM_SHARED_TC']!='1':
    raise SystemExit('Shared NIC: verify recovery and explicitly set E2_CONFIRM_SHARED_TC=1')
PY
fi

# Match metadata, not just the OFF/AWARE directory prefix. Old v3 and short
# smoke tests must never count as completed three-hour v4/ring experiments.
existing() {
    python3 - "$1" "$2" "$3" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
method,transport,mode=sys.argv[1:]
root=Path('ddp' if method=='ddp' else 'diloco')/'results'
profile='E3_native_nccl_frontend_control_v4' if transport=='nccl' else 'E3_decentralized_tcp_ring_v1'
sha=hashlib.sha256(Path(os.environ['AUTO_TRACE']).read_bytes()).hexdigest()
def load(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
def same(a,b):
    try:return float(a)==float(b)
    except (ValueError,TypeError):return a==b
def matches_dict(actual,keys):
    return all(same(actual.get(k),os.environ[k]) for k in keys)
ring_keys='E3_RING_PORT E3_RING_FRAME_BYTES E3_RING_MARGIN_S E3_RING_FACTOR E3_RING_OPEN_GUARD_S E3_RING_MIN_TRANSFER_S E3_RING_CONNECT_TIMEOUT_S E3_RING_DRAIN_GUARD_S E3_RING_DRAIN_TIMEOUT_S'.split()
batch_keys='E3_GRANT_MAX_OPS E3_LOCAL_STEP_BUDGET_S E3_OBSERVED_FACTOR E3_MIN_TRANSFER_S E3_OBSERVED_WINDOW E3_OBSERVED_MAX_AGE_S E3_OBSERVED_QUANTILE E3_CONTACT_OPEN_GUARD_S E3_LAUNCH_GUARD_S E3_CONTROL_POLL_S E3_RENDEZVOUS_TIMEOUT'.split()
rep=os.environ['AUTO_REPETITION']
for d in sorted(root.glob(f'E3_1_{method}_{mode}_*'),reverse=True):
    if not d.is_dir():continue
    import re
    tag=re.search(r'(?:^|_)R(\d+)(?:_|$)',d.name)
    if (tag and 'R'+tag[1]!=rep) or (not tag and rep!='R01'):continue
    c=load(d/'e2_run_config.json')
    if not c:
        # Preserve failed preflight attempts; only skip if named by this exact campaign.
        label=('nccl_K32' if mode=='aware' else 'nccl_native') if transport=='nccl' else 'ring_frame1MiB'
        if os.environ['AUTO_RETRY_FAILED']=='0' and f'_{label}_{os.environ["MAX_SECONDS"]}s_{rep}_' in d.name:
            print(str(d)+' (existing incomplete attempt; no config)');break
        continue
    wanted=dict(method=method,schedule_mode=mode,training_profile=profile,
        duration_s=int(os.environ['MAX_SECONDS']),dataset=os.environ['DATASET'],
        trace_sha256=sha,tc_applied=True,rate_scale=1,seed=2026,max_epochs=100,
        model='ResNet18_CIFAR_stem',batch_size=32,num_workers=4,
        comm_timeout_s=900,shutdown_grace_s=300,eval_interval_s=300,
        resource_interval_s=2,outer_lr=.3,outer_momentum=.9,
        e3_margin_s=15,e3_estimate_factor=2)
    if any(not same(c.get(k),v) for k,v in wanted.items()):continue
    if transport=='ring':
        if not matches_dict(c.get('ring_config') or {},ring_keys):continue
    else:
        if c.get('nccl_algorithm')!='Ring' or c.get('control_transport')!='frontend_tcp' or c.get('nccl_topology_policy')!='ring_auto':continue
        if mode=='aware' and not matches_dict(c.get('nccl_grant_config') or {},batch_keys):continue
        if mode=='off' and not matches_dict(c.get('reactive_stop_config') or {},['E3_REACTIVE_STOP_CHECK_OPS','E3_REACTIVE_STOP_TIMEOUT_S']):continue
    status=load(d/'execution_status.json')
    summary=load(d/'e3_summary.json')
    success=status.get('success') is True and summary.get('run_success') is True
    if success or os.environ['AUTO_RETRY_FAILED']=='0':
        print(str(d)+(' (success)' if success else ' (failed/incomplete; preserved)'));break
PY
}

# Fail closed if remaining reservation time cannot be established.
remaining() {
    python3 - <<'PY'
import datetime,json,os,subprocess,time
override=os.getenv('AUTO_JOB_END_UNIX')
if override:
    print(int(float(override)-time.time()));raise SystemExit
j=os.environ['OAR_JOB_ID']
try:
    raw=json.loads(subprocess.check_output(['oarstat','-fj',j,'-J'],text=True,timeout=20))
    obj=raw.get(j,raw) if isinstance(raw,dict) else raw[0]
    if obj.get('state')!='Running':raise ValueError('job not Running')
    start=obj.get('start_time') or obj.get('startTime')
    wall=obj.get('walltime') or obj.get('wall_time')
    if isinstance(wall,str) and ':' in wall:
        h,m,s=map(int,wall.split(':'));wall=h*3600+m*60+s
    print(int(float(start)+float(wall)-time.time()))
except Exception as e:
    raise SystemExit(f'Cannot determine OAR deadline: {e}; set AUTO_JOB_END_UNIX to verified job end epoch')
PY
}

# Read-only check. Never remove a qdisc automatically or ignore surviving jobs.
safe_nodes() {
    python3 - <<'PY'
import re,shlex
from remote import read_state,ssh
for n in read_state()['nodes']:
    h=n['host']; iface=shlex.quote(n['iface'])
    q=ssh(h,'tc qdisc show dev '+iface,timeout=25)
    if re.search(r'^qdisc htb 1: root\b',q,re.M):
        raise SystemExit(f'BLOCKED: residual qdisc on {h}; inspect ./cleanup.sh')
    # Match actual Python worker processes, not the SSH command itself.
    cmd="ps -eo comm=,args= | awk '$1 ~ /^python/ && $0 ~ /synthesized_E3_1\\/.*(tc_agent|ring_train|nccl_train|\\/train)\\.py/ {print}'"
    active=ssh(h,cmd,timeout=25)
    if active.strip():raise SystemExit(f'BLOCKED: worker still active on {h}: {active}')
    print('Clean:',h)
PY
}
interrupted=0
trap 'interrupted=1' INT TERM
failed=0
skipped=0
launched=0
for method in ddp diloco8 diloco64; do
    for transport in nccl ring; do
        for mode in off aware; do
            label=ring_frame1MiB
            if [[ "$transport" == nccl ]]; then
                label=nccl_native
                [[ "$mode" == aware ]] && label=nccl_K32
            fi
            found=$(existing "$method" "$transport" "$mode")
            if [[ -n "$found" ]]; then
                echo "SKIP $method $transport $mode: $found"
                skipped=$((skipped+1))
                continue
            fi
            pair="${label}_${MAX_SECONDS}s_${AUTO_REPETITION}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
            launcher=./diloco/run.sh
            [[ "$method" == ddp ]] && launcher=./ddp/run.sh
            echo "NEXT $method $transport $mode; pair=$pair"
            if [[ "$AUTO_DRY_RUN" == 1 ]]; then continue; fi
            left=$(remaining)
            need=$((MAX_SECONDS+AUTO_START_BUFFER_S+E2_SHUTDOWN_GRACE))
            if (( left < need )); then
                echo "PAUSE: reservation remaining ${left}s; need ${need}s. Resume in another prepared reservation."
                exit 3
            fi
            safe_nodes
            if (( interrupted )); then exit 130; fi
            launched=$((launched+1))
            if METHOD="$method" CASE="$mode" E3_SYNC_TRANSPORT="$transport" E3_PAIR_ID="$pair" "$launcher" --trace "$TRACE" --rate-scale 1; then
                echo "FINISHED $method $transport $mode"
            else
                rc=$?
                failed=$((failed+1))
                echo "FAILED ($rc) $method $transport $mode; preserving results"
                # Interruption must not start another experiment.
                if (( rc >= 128 )); then exit "$rc"; fi
            fi
            if (( interrupted )); then echo 'Interrupted; stopping campaign'; exit 130; fi
            safe_nodes
        done
    done
done
echo "Campaign finished: launched=$launched skipped=$skipped failed=$failed"
(( failed == 0 ))
