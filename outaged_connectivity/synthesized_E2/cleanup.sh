#!/usr/bin/env bash
# Read-only by default. Explicit E2_CONFIRM_CLEANUP=1 removes 1: HTB root qdisc.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export E2_REMOTE_MODE=deploy
python3 - <<'PY'
import os
import re
import shlex
import sys
from remote import read_state, ssh, privileged

state = read_state()
confirm = os.getenv("E2_CONFIRM_CLEANUP") == "1"
failed = False

for node in state["nodes"]:
    host = node["host"]
    iface = node["iface"]
    qiface = shlex.quote(iface)

    print(f"\n========== {host} ==========", flush=True)

    qdiscs = ssh(host, f"tc qdisc show dev {qiface}")
    print(qdiscs)

    owned_shape = re.search(
        r"^qdisc htb 1: root\b",
        qdiscs,
        re.MULTILINE
    )

    if not owned_shape:
        print("No E2-style root qdisc detected.")
        continue

    # This is only an E2-style signature, not proof of ownership.
    active = ssh(
        host,
        "pgrep -af '[t]c_agent.py' || true"
    )

    if active.strip():
        print("ERROR: A tc_agent.py process is still running:")
        print(active)
        failed = True
        continue

    if not confirm:
        print("DRY RUN: root HTB qdisc detected.")
        print("Check its ownership before authorizing cleanup.")
        continue

    print("WARNING: Deleting HTB 1: root qdisc.")
    try:
        privileged(host, f"tc qdisc del dev {qiface} root")
        print("Cleanup completed.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        failed = True

sys.exit(1 if failed else 0)
PY
