#!/usr/bin/env bash
# Sequentially run seven training methods in the same four-host OAR reservation.
# 7 x 1h E2 experiments exceed a normal 4h reservation, so pick shorter pilots
# or reserve a longer walltime. Resumes are not automatic.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ "${E2_CONFIRM_ALL:-0}" == 1 ]] || { echo 'Set E2_CONFIRM_ALL=1 after reserving enough walltime' >&2; exit 2; }
for m in ddp gossip8 gossip64 diloco8 diloco64 fl8 fl64;do
    echo "===== E2 $m ====="
    METHOD="$m" "$HERE/run.sh" "$@"
done
