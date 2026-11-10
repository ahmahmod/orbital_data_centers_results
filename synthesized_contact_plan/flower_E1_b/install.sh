#!/usr/bin/env bash
# Copy the exact completed E1_a_new instrumentation into E1_b without touching E1_a_new.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REF="${E1_A_NEW_DIR:-$HERE/../E1_a}"
[[ -f "$REF/train.py" && -f "$REF/plot.py" && -f "$REF/socket_counter_watch.py" ]] || {
  echo "Missing E1_a_new at $REF (set E1_A_NEW_DIR=/path/to/E1_a_new)" >&2; exit 1;
}
cp -p "$REF/train.py" "$HERE/e1a_reference.py"
for name in plot.py socket_counter_watch.py analyze_socket_counters.py socket_monitor_health.py plot_tcp_counters.py nccl_diagnostics.py selftest_rootless.py; do
  [[ -f "$REF/$name" ]] && cp -p "$REF/$name" "$HERE/$name"
done
printf '%s\n' 'Copied E1_a_new data, meter schema and rootless measurement/plotting tools into E1_b.'
