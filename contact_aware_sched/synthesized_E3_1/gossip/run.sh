#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${METHOD:=gossip64}"
case "$METHOD" in gossip8|gossip64) ;; *) echo "Use METHOD=gossip8 or gossip64" >&2;exit 2;;esac
export METHOD
exec "$HERE/../run_tech.sh" "$@"
