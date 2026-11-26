#!/usr/bin/env bash
# Reuse E2 network state only when it belongs to the SAME active OAR reservation.
# Python and CIFAR data now live at the repository root for every experiment.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
E2="${E2_SOURCE_DIR:-$ROOT/outaged_connectivity/synthesized_E2}"
[[ -d "$E2" ]] || { echo "E2 directory absent: $E2" >&2;exit 2; }
[[ -n "${OAR_JOB_ID:-}" ]] || { echo 'Active OAR reservation required' >&2; exit 2; }
python3 - "$E2/kavlan_state.json" "$OAR_JOB_ID" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if str(s.get('oar_job_id'))!=sys.argv[2]:
    raise SystemExit('E2 network state belongs to another OAR job: run E3 setup.sh')
names=sorted(n['host'].split('.')[0] for n in s['nodes'])
if names!=['graffiti-5','graffiti-7','graffiti-8','graffiti-9']:
    raise SystemExit('E2 state does not contain the four specified Graffiti nodes')
print('Validated E2 network state in current reservation',sys.argv[2])
PY
if [[ ! -e "$HERE/kavlan_state.json" && ! -L "$HERE/kavlan_state.json" ]];then
  ln -s "$E2/kavlan_state.json" "$HERE/kavlan_state.json"
elif [[ "$(realpath -e "$HERE/kavlan_state.json" 2>/dev/null || true)" != "$(realpath -e "$E2/kavlan_state.json")" ]];then
  echo "ERROR: E3 state exists but does not match E2; will not overwrite" >&2;exit 2
fi
[[ -x "$ROOT/.venv/bin/python" ]] || { echo "Shared $ROOT/.venv missing; run shared_runtime.sh on allocated GPU node" >&2;exit 2; }
[[ -d "$ROOT/data/cifar-100-python" ]] || { echo "Shared CIFAR data missing; run shared_runtime.sh" >&2;exit 2; }
echo 'Ready: shared root runtime and verified same-job E2 network state.'
