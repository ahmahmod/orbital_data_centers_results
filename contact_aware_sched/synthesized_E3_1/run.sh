#!/usr/bin/env bash
# Launch E2 only after GPU, Python and reservation verification.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export E2_REMOTE_MODE=deploy
: "${METHOD:=ddp}"
: "${DATASET:=synthetic}"
: "${MAX_SECONDS:=180}"
: "${STARTUP_LEAD:=150}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
export E2_PYTHON="${E2_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -n "${OAR_JOB_ID:-}" ]] || { echo 'ERROR: Run inside active OAR reservation' >&2;exit 2; }
[[ -f "$HERE/kavlan_state.json" ]] || { echo 'ERROR: Run E2_CONFIRM_NETWORK=1 ./setup.sh first' >&2;exit 2; }
[[ -x "$E2_PYTHON" ]] || { echo "ERROR: Missing shared environment: $E2_PYTHON; run E2_CONFIRM_NETWORK=1 ./setup.sh" >&2; exit 2; }
# Check that network_state.json really belongs to the current job.
MODE="$(python3 - "$HERE/kavlan_state.json" "$OAR_JOB_ID" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert str(s.get('oar_job_id'))==sys.argv[2], 'state belongs to a different OAR reservation'
assert len(s['nodes'])==4
print(s.get('network_mode','kavlan'))
PY
)"

# ==========================================================
# Preflight: reject leftover E2 qdiscs on allocated nodes
# ==========================================================

if [[ "${TC_APPLY:-0}" == 1 ]]; then
  echo "Checking traffic-control state on all four nodes..."

  python3 - <<'PY'
import re
import sys
from remote import read_state, ssh

state = read_state()
failures = []

for node in state["nodes"]:
    host = node["host"]
    iface = node["iface"]

    output = ssh(host, f"tc qdisc show dev {iface}")
    root = [
        line for line in output.splitlines()
        if re.match(r"^qdisc htb 1: root\b", line)
    ]

    if root:
        failures.append((host, iface))
        print(f"BLOCKED: {host} has an existing HTB 1: root qdisc")
    else:
        print(f"PASS: {host} — no existing E2 root qdisc")

if failures:
    print("\nERROR: Existing qdiscs detected.")
    print("Inspect running processes and use ./cleanup.sh before retrying.")
    sys.exit(1)

print("PASS: Traffic-control preflight completed.")
PY
fi


# Do not enable root qdisc replacement without explicit informed confirmation.
ARGS=(--method "$METHOD" --max-seconds "$MAX_SECONDS" --dataset "$DATASET" --startup-lead "$STARTUP_LEAD" --python "$E2_PYTHON" --schedule-mode "${E3_SCHEDULE_MODE:-off}" --e3-margin-s "${E3_MARGIN_S:-15}" --e3-estimate-factor "${E3_ESTIMATE_FACTOR:-2}")
if [[ "${TC_APPLY:-0}" == 1 ]]; then
  [[ "${E2_CONFIRM_SHAPING:-0}" == 1 ]] || { echo 'ERROR: set E2_CONFIRM_SHAPING=1 before tc replay' >&2;exit 2; }
  if [[ "$MODE" == selective_tc ]]; then
    [[ "${E2_CONFIRM_SHARED_TC:-0}" == 1 ]] || { echo 'ERROR: selective_tc modifies MANAGEMENT NIC. Set E2_CONFIRM_SHARED_TC=1 after verifying recovery.' >&2;exit 2; }
  fi
  ARGS+=(--apply)
else
  echo 'DRY RUN: no tc qdisc changes; satellite trace is logged only.'
fi
exec python3 "$HERE/runner.py" "${ARGS[@]}" "$@"
