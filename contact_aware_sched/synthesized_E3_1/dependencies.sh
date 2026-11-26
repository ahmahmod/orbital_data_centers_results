#!/usr/bin/env bash
# Compatibility entry point: E2 and E3 install from the common repository root.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
exec bash "$ROOT/shared_runtime.sh"
