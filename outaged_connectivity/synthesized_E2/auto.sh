#!/usr/bin/env bash
# Unattended E2 CIFAR-100 seven-method benchmark.
# Invoke ONCE inside an already active four-node OAR reservation: ./auto.sh
# Applies the contact trace. It NEVER deletes an unknown/leftover qdisc or kills
# processes that the coordinator failed to clean up. Unsafe state stops the batch.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[[ "$HERE" != / && -n "$HERE" ]] || { echo 'ERROR: invalid project path' >&2; exit 2; }
export E2_REMOTE_MODE=deploy
export E2_PYTHON="${E2_PYTHON:-$HERE/.venv/bin/python}"
# Explicit experiment configuration. No interactive confirmations required.
export DATASET=cifar100 TC_APPLY=1 E2_CONFIRM_SHAPING=1 E2_CONFIRM_SHARED_TC=1
export E2_COMM_TIMEOUT=1200 E2_EVAL_INTERVAL_S=300 E2_RESOURCE_INTERVAL_S=2
export E2_LOG_EVERY=20 STARTUP_LEAD=150
export E2_FLOWER_ROUNDS="${E2_FLOWER_ROUNDS:-500}"
DURATION="${DURATION:-10800}"
METHODS=(ddp gossip8 gossip64 diloco8 diloco64 fl8 fl64)
# METHODS=(ddp gossip8 gossip64 diloco8 diloco64)
# METHODS=(ddp gossip8 gossip64 diloco8 diloco64)
RESULTS="$HERE/results"

# Any output directory counts as attempted, even incomplete/failed results.
# Rename or remove an incomplete directory to retry it.
experiment_exists() {
  local method="$1"
  # Only skip a matching duration/dataset/trace. Old 1-hour directories
  # must NOT cause a 3-hour benchmark to be skipped.
  "$E2_PYTHON" - "$RESULTS" "$method" "$DURATION" "$DATASET" "$HERE/traces/directed_trace_4workers.csv" "$E2_FLOWER_ROUNDS" <<'PY'
import hashlib, json, pathlib, sys
root, method, duration, dataset, trace, rounds = sys.argv[1:]
fingerprint = hashlib.sha256(pathlib.Path(trace).read_bytes()).hexdigest()
for folder in pathlib.Path(root).glob(f'E2_{method}_*'):
    if not folder.is_dir():
        continue
    try:
        config = json.loads((folder / 'e2_run_config.json').read_text())
    except (OSError, ValueError):
        continue
    if (config.get('method') == method
        and int(config.get('duration_s', -1)) == int(duration)
        and config.get('dataset') == dataset
        and config.get('trace_sha256') == fingerprint
        and config.get('tc_applied') is True
        and (not method.startswith('fl') or int(config.get('flower_rounds') or -1) == int(rounds))):
        print(f'Matching prior attempt: {folder}')
        sys.exit(0)
sys.exit(1)
PY
}
[[ "$E2_FLOWER_ROUNDS" =~ ^[0-9]+$ ]] && (( E2_FLOWER_ROUNDS >= 1 )) ||
  { echo 'E2_FLOWER_ROUNDS must be a positive integer' >&2; exit 2; }
[[ "$DURATION" =~ ^[0-9]+$ ]] && (( DURATION >= 60 && DURATION <= 18000 )) ||
  { echo 'DURATION must be an integer 60..18000' >&2; exit 2; }

err() { echo "ERROR: $*" >&2; exit 2; }
[[ -n "${OAR_JOB_ID:-}" && -r "${OAR_NODEFILE:-/nonexistent}" ]] || err 'Start from an active four-node OAR job.'
[[ -x "$HERE/run.sh" && -x "$E2_PYTHON" ]] || err 'run.sh or the E2 Python environment is missing.'
[[ -r "$HERE/kavlan_state.json" ]] || err 'Run setup.sh in this OAR reservation first.'
[[ -r "$HERE/traces/directed_trace_4workers.csv" ]] || err 'Contact trace is missing.'
# First-run CIFAR download is automatic (before any traffic shaping).
if [[ ! -d "$HERE/data/cifar-100-python" ]]; then
  echo 'CIFAR-100 missing; downloading training and test sets before the overnight batch...'
  "$E2_PYTHON" "$HERE/download_cifar.py" || err 'CIFAR-100 download failed; no experiment started.'
fi
[[ -d "$HERE/data/cifar-100-python" ]] || err 'CIFAR-100 download did not create the expected directory.' 

# Prevent two batch coordinators running against the same shared E2 state.
exec 9>"$HERE/.e2_auto.lock"
flock -n 9 || err 'Another auto.sh batch already owns this project.'

# Validates current OAR allocation and absence of active E2 scripts/HTB qdiscs.
# This is read-only, never removes shared-NIC qdiscs automatically.
check_network_clean() {
  python3 - "$HERE" <<'PY'
import os, pathlib, re, shlex, sys
from remote import read_state, ssh
here=pathlib.Path(sys.argv[1])
state=read_state()
assert len(state['nodes']) == 4, 'Expected four workers'
if state.get('network_mode') not in ('selective_tc','kavlan'):
    raise RuntimeError('Unsupported network mode')
probe=r'''import os, pathlib
me=os.getpid(); targets={'train.py','train_flower.py','tc_agent.py'}
hits=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit() or int(d.name)==me: continue
 try:
  args=(d/'cmdline').read_bytes().split(b'\0')
  if any(pathlib.PurePosixPath(a.decode(errors='replace')).name in targets for a in args if a):
   hits.append(d.name+':'+b' '.join(args).decode(errors='replace')[:180])
 except (OSError,PermissionError): pass
print('\n'.join(hits))'''
errors=[]
for n in state['nodes']:
    host,iface=n['host'],n['iface']
    try:
        # SSH runs on management path; if unavailable, fail closed.
        active=ssh(host,'python3 -c '+shlex.quote(probe),timeout=25)
        if active.strip(): errors.append(f'{host}: active E2 workers: {active}')
        q=ssh(host,'tc qdisc show dev '+shlex.quote(iface),timeout=25)
        if re.search(r'^qdisc htb 1: root\b',q,re.M):
            errors.append(f'{host}: leftover root HTB qdisc on {iface}')
        marker='/run/e2_tc_'+iface+'.json'
        marker_state=ssh(host,'if test -e '+shlex.quote(marker)+'; then echo FOUND; fi',timeout=25)
        if marker_state.strip(): errors.append(f'{host}: leftover ownership marker {marker}')
        # Check availability of the same data directory on every worker.
        ssh(host,'test -d '+shlex.quote(str(here/'data/cifar-100-python')),timeout=25)
    except Exception as exc:
        errors.append(f'{host}: cannot verify clean state/dataset: {exc}')
if errors:
    print('UNSAFE STATE:', '\n - '.join(errors),sep='\n - ',file=sys.stderr)
    sys.exit(10)
print('PASS: four workers reachable, CIFAR-100 available, no E2 workers/qdiscs/markers',flush=True)
PY
}

# Wait for coordinator's normal cleanup to settle, then fail closed.
wait_for_clean() {
  local attempts=12
  for ((i=1; i<=attempts; i++)); do
    if check_network_clean; then return 0; fi
    echo "Cleanup still incomplete ($i/$attempts); checking again in 10 seconds..." >&2
    sleep 10
  done
  return 1
}

# OAR records are provider/version dependent. When a parseable job end time is
# available, stop before starting a run that cannot fit; no guessed extension.
remaining_oar_seconds() {
  python3 - "${OAR_JOB_ID}" <<'PY'
import datetime,json,subprocess,sys,time
jobid=sys.argv[1]
try:
    p=subprocess.run(['oarstat','-J','-j',jobid],capture_output=True,text=True,timeout=15,check=True)
    raw=json.loads(p.stdout)
    obj=raw.get(jobid,raw) if isinstance(raw,dict) else None
    if not isinstance(obj,dict): raise ValueError('unknown OAR JSON')
    wall=obj.get('walltime')
    start=obj.get('start_time') or obj.get('startTime')
    if wall is None or not start: raise ValueError('missing walltime or start_time')
    if isinstance(wall,str) and ':' in wall:
        pieces=[int(x) for x in wall.split(':')]
        wall=pieces[0]*3600+pieces[1]*60+pieces[2]
    else: wall=int(wall)
    if isinstance(start,str) and not start.isdigit():
        start=datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp()
    start=float(start)
    if start < 1_000_000_000 or wall <= 0: raise ValueError('not an epoch timestamp / duration')
    print(max(0,int(start+wall-time.time())))
except Exception:
    print('unknown')
PY
}

batch="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS"
summary="$RESULTS/E2_batch_${batch}.tsv"
log="$RESULTS/E2_batch_${batch}.log"
exec > >(tee -a "$log") 2>&1
printf 'method\trun_status\trunner_exit\tclean_after\tstart_utc\tend_utc\toutput_dir\n' > "$summary"
echo "E2 overnight batch: $batch"
echo "Methods: ${METHODS[*]}"
echo "Flower: ${E2_FLOWER_ROUNDS} rounds per FL method, with global-model evaluation on all four workers every round."
echo "CIFAR-100; contact shaping ON; up to ${DURATION}s per run (early exit after all trainers finish); NCCL/Flower timeout ${E2_COMM_TIMEOUT}s"
echo "Batch log: $log"
echo "Current OAR job: $OAR_JOB_ID"

# Only require enough time for the next missing experiment.
SHUTDOWN_GRACE="${E2_SHUTDOWN_GRACE:-300}"
[[ "$SHUTDOWN_GRACE" =~ ^[0-9]+$ ]] && (( SHUTDOWN_GRACE >= 60 )) || err 'E2_SHUTDOWN_GRACE must be >= 60'
required=$(( DURATION + STARTUP_LEAD + SHUTDOWN_GRACE + 240 ))
remaining="$(remaining_oar_seconds)"
echo "Minimum time for one experiment: ${required}s"
[[ "$remaining" != unknown ]] || err 'Cannot determine OAR remaining walltime.'
echo "OAR time remaining: ${remaining}s"

check_network_clean || err 'Initial state is not clean; no experiment started.'
completed=0
failed=0
skipped=0
not_started=0
for method in "${METHODS[@]}"; do
  if experiment_exists "$method"; then
    echo "SKIP: $method already has an output directory"
    skipped=$((skipped + 1))
    continue
  fi
  remaining="$(remaining_oar_seconds)"
  if [[ "$remaining" == unknown ]]; then
    echo "STOP: OAR remaining walltime unknown" >&2
    not_started=$(( ${#METHODS[@]} - skipped - completed - failed ))
    break
  fi
  if (( remaining < required )); then
    echo "STOP: ${remaining}s remain; $method needs ${required}s" >&2
    not_started=$(( ${#METHODS[@]} - skipped - completed - failed ))
    break
  fi
  echo
  echo "===== [$((completed + failed + skipped + 1))/${#METHODS[@]}] $method ====="
  begin="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  out="$RESULTS/E2_${method}_${batch}"
  rc=0
  METHOD="$method" MAX_SECONDS="$DURATION" "$HERE/run.sh" --out "$out" || rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  # Even an unsuccessful training attempt can be followed by another, but ONLY
  # when all four remote nodes are checked clean. Never delete shared NIC rules.
  clean=YES
  if ! wait_for_clean; then clean=NO; fi
  status=FAIL
  if ((rc == 0)) && [[ "$clean" == YES ]]; then
    status=PASS; completed=$((completed+1))
  else
    failed=$((failed+1))
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$method" "$status" "$rc" "$clean" "$begin" "$end" "$out" >> "$summary"
  # runner.py now finalizes as soon as all trainers exit successfully;
  # the contact replay's full DURATION is only an upper time limit.
  completion_reason="$("$E2_PYTHON" - "$out/execution_status.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    obj = json.loads(p.read_text())
    print(obj.get('completion_reason', 'unspecified'))
except (OSError, ValueError):
    print('status_unavailable')
PY
)"
  echo "Completed $method: runner_exit=$rc; clean_after=$clean; status=$status; reason=$completion_reason"
  if [[ "$status" == PASS ]]; then
    echo "NEXT: $method finalized; advancing to the next training method."
  fi
  if [[ "$clean" != YES ]]; then
    echo 'SAFETY STOP: leftover processes or traffic-control state detected. No subsequent methods launched.' >&2
    break
  fi
  if ((rc != 0)); then
    echo 'NOTE: experiment failed but cleanup verified. Continuing to the next independent method.'
  fi
  sleep 10
 done

echo
printf 'BATCH SUMMARY: %s PASS, %s FAIL, %s SKIPPED, %s NOT STARTED, %s TOTAL\n' "$completed" "$failed" "$skipped" "$not_started" "${#METHODS[@]}"
cat "$summary"
echo "Results stored under $RESULTS; plotting runs during runner.py finalization."
if (( failed > 0 )); then exit 1; fi
if (( not_started > 0 )); then
  echo 'Remaining experiments can run in a later OAR reservation.'
  exit 3
fi
echo 'All configured methods have output directories.'
exit 0
