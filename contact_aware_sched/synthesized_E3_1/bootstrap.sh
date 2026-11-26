#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export E2_REMOTE_MODE=deploy
exec python3 "$HERE/bootstrap.py" --deploy --apply "$@"
