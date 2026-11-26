#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${METHOD:=fl64}"
case "$METHOD" in fl8|fl64) ;; *) echo "Use METHOD=fl8 or fl64" >&2;exit 2;;esac
export METHOD
exec "$HERE/../run_tech.sh" "$@"
