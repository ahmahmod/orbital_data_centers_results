#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export E2_REMOTE_MODE=deploy
python3 - <<'PY'
from remote import hosts,ssh,profile,port_ref
print('KaVLAN inspection — remote mode is controlled by E2_REMOTE_MODE')
for h in hosts():
 p=profile(h)
 print('===',h,'===',p['gpu'],p['iface'],port_ref(h))
 print(ssh(h,'hostname; ip -br addr; ip -4 route; nvidia-smi -L | head -4'))
PY
