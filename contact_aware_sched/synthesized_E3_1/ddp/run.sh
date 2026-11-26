#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD=ddp
exec "$HERE/../run_tech.sh" "$@"
