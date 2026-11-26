#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${METHOD:=diloco64}"
case "$METHOD" in diloco8|diloco64) ;; *) echo "Use METHOD=diloco8 or diloco64" >&2; exit 2;;esac
export METHOD
export E2_OUTER_LR="${E2_OUTER_LR:-0.3}"
export E2_OUTER_MOMENTUM="${E2_OUTER_MOMENTUM:-0.9}"
exec "$HERE/../run_tech.sh" "$@"
