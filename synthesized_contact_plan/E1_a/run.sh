#!/usr/bin/env bash
# E1_a synchronous DDP, Grid'5000/OAR, arbitrary hosts x GPUs per host.
# Defaults to four hosts x one GPU; arbitrary host/GPU counts supported.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
log(){ printf '\n========== %s =========\n' "$*"; }
die(){ echo "ERROR: $*" >&2; exit 1; }
: "${N_NODES:=4}"
: "${GPUS_PER_NODE:=1}"
: "${MAX_SECONDS:=3600}"
: "${EPOCHS:=100}"
: "${SEED:=2026}"
: "${BATCH_SIZE:=32}"
: "${NUM_WORKERS:=4}"
: "${LOG_EVERY_STEPS:=10}"
: "${COLLECTIVE_TRACE_EVERY:=1}"
: "${SAMPLE_PERIOD_S:=1}"
: "${DATA_ROOT:=$HERE/../data}"
: "${PYTHON:=$HERE/../.venv311/bin/python}"
: "${MASTER_PORT:=29527}"
: "${RUN_ID:=E1_a_${N_NODES}hosts_${GPUS_PER_NODE}gpus_seed${SEED}_$(date -u +%Y%m%dT%H%M%S%NZ)}"
: "${OUT:=$HERE/results/$RUN_ID}"
: "${NCCL_DEBUG:=INFO}"
: "${NCCL_SOCKET_IFNAME:=br0}"
: "${NCCL_SOCKET_FAMILY:=AF_INET}"
: "${NET_IFACE:=br0}"
: "${GPU_IDLE_THRESHOLD:=20}"
: "${SHOW_PROGRESS:=1}"
# Rootless ss -tinp counter sampling. Legacy capture flags are accepted as aliases.
: "${MEASURE_TCP:=1}"
: "${TCP_SAMPLE_PERIOD_S:=0.1}"
: "${CAPTURE_ALL_PEERS:=0}"
: "${CAPTURE_SOCKETS:=0}"
: "${NCCL_TRANSPORT_MODE:=native}"
: "${NCCL_IB_DISABLE:=1}"  # Explicit TCP-based measurement configuration; alters native RDMA performance.
: "${MONITOR_DURATION_S:=$((MAX_SECONDS+180))}"
[[ "$NCCL_TRANSPORT_MODE" == native || "$NCCL_TRANSPORT_MODE" == tcp_observable ]] || die 'Invalid NCCL_TRANSPORT_MODE'
if [[ "$NCCL_TRANSPORT_MODE" == tcp_observable ]]; then
  NCCL_P2P_DISABLE=1; NCCL_SHM_DISABLE=1; NCCL_IB_DISABLE=1
fi
if (( CAPTURE_ALL_PEERS || CAPTURE_SOCKETS )); then
  echo 'NOTE: CAPTURE_* flags are legacy; tcpdump removed. Enabling rootless TCP sampling.' >&2
  MEASURE_TCP=1
fi
[[ -x "$PYTHON" ]] || die "Missing reusable Python environment: $PYTHON (set PYTHON=/shared/.../bin/python)"
[[ -d "$DATA_ROOT/cifar-100-python" ]] || die "Missing reusable CIFAR100 dataset: $DATA_ROOT/cifar-100-python (set DATA_ROOT=...)"
(( N_NODES >= 1 && GPUS_PER_NODE >= 1 && N_NODES * GPUS_PER_NODE >= 2 )) || die "Require >=2 DDP ranks (N_NODES>=1 and GPUS_PER_NODE>=1)"
NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
[[ -r "$NODEFILE" ]] || die "Run inside active OAR reservation (OAR_NODEFILE missing)"
mapfile -t NODES < <(sort -u "$NODEFILE")
(( ${#NODES[@]} == N_NODES )) || die "Expected $N_NODES hosts, allocated ${#NODES[@]}: ${NODES[*]}"
command -v oarsh >/dev/null || die "oarsh missing"
MASTER_ADDR="${MASTER_ADDR:-${NODES[0]}}"
[[ ! -e "$OUT" ]] || die "Output directory already exists: $OUT; use a new RUN_ID"
mkdir -p "$OUT";OUT="$(realpath "$OUT")";DATA_ROOT="$(realpath "$DATA_ROOT")"
log "Check existing GPU/CIFAR/Python resources"
for node in "${NODES[@]}"; do
  echo "Node $node"
  oarsh "$node" "test -x '$PYTHON' && '$PYTHON' -c 'import torch,torchvision,numpy,matplotlib; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.device_count())'" || die "CUDA/Python unavailable on $node"
  count=$(oarsh "$node" "nvidia-smi -L | wc -l" | tr -d '[:space:]')
  (( count >= GPUS_PER_NODE )) || die "$node: $count GPUs detected, need $GPUS_PER_NODE"
  oarsh "$node" "test -d '$DATA_ROOT/cifar-100-python'" || die "Dataset not visible on $node"
  oarsh "$node" "test -d '$HERE' && test -w '$OUT'" || die "Project/output not shared across $node"
done
log "E1_a starts: ${NODES[*]}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Workers: $(( N_NODES * GPUS_PER_NODE )) | output: $OUT"
echo "NCCL: interface $NCCL_SOCKET_IFNAME, IPv4 family $NCCL_SOCKET_FAMILY | mode: $NCCL_TRANSPORT_MODE"
# No network shaping here: E1_a is stable-network baseline.
pids=();tail_pid=''
cleanup(){
  local code=$?
  trap - EXIT
  if [[ -n "${tail_pid:-}" ]]; then kill "$tail_pid" 2>/dev/null || true; wait "$tail_pid" 2>/dev/null || true; fi
  if (( code != 0 )); then
    echo "E1_a failed (exit $code). Inspect $OUT/node_*.log" >&2
    for log_file in "$OUT"/node_*.log; do
      [[ -f "$log_file" ]] || continue
      echo "===== $log_file =====" >&2
      tail -25 "$log_file" >&2 || true
    done
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
# Discover the routable IP for EACH allocated host, not only a single peer.
# Also used by workers to record the capture-interface IP in worker_map.json.
HOST_IPS=()
for (( ci=0; ci<N_NODES; ci++ )); do
  node="${NODES[$ci]}"
  ip=$(oarsh "$node" "ip -4 -o addr show dev '$NET_IFACE' | grep -oE 'inet [0-9.]+' | head -1 | cut -d' ' -f2" | tr -d '\r')
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "No IPv4 on $NET_IFACE for $node (got '$ip')"
  HOST_IPS+=("$ip")
done
if (( MEASURE_TCP )); then
  log "Rootless TCP monitoring: started by local rank 0 inside each torchrun host"
  # No extra oarsh monitor jobs and no shared monitor.stop marker. Monitoring must
  # not prevent training from starting if ss/process ownership is restricted.
  for node in "${NODES[@]}"; do
    if ! oarsh "$node" "command -v ss >/dev/null"; then
      echo "WARNING: ss unavailable on $node: training will continue, TCP metrics may be unavailable" >&2
    fi
  done
fi

for (( i=0; i<N_NODES; ++i )); do
  node="${NODES[$i]}"
  printf -v remote 'cd %q && export OMP_NUM_THREADS=1 NCCL_DEBUG=%q NCCL_SOCKET_IFNAME=%q NCCL_SOCKET_FAMILY=%q NCCL_IB_DISABLE=%q E1_WORKER_IP=%q E1_TCP_MEASURE=%q E1_TCP_PERIOD=%q E1_TCP_DURATION=%q && %q -m torch.distributed.run --nnodes=%q --nproc-per-node=%q --node-rank=%q --master-addr=%q --master-port=%q %q --output=%q --data-root=%q --epochs=%q --batch-size=%q --num-workers=%q --seed=%q --max-seconds=%q --log-every-steps=%q --collective-trace-every=%q --sample-period-s=%q --gpu-idle-threshold=%q --net-iface=%q' \
    "$HERE" "$NCCL_DEBUG" "$NCCL_SOCKET_IFNAME" "$NCCL_SOCKET_FAMILY" "$NCCL_IB_DISABLE" "${HOST_IPS[$i]}" "$MEASURE_TCP" "$TCP_SAMPLE_PERIOD_S" "$MONITOR_DURATION_S" \
    "$PYTHON" "$N_NODES" "$GPUS_PER_NODE" "$i" "$MASTER_ADDR" "$MASTER_PORT" \
    "$HERE/train.py" "$OUT" "$DATA_ROOT" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "$MAX_SECONDS" \
    "$LOG_EVERY_STEPS" "$COLLECTIVE_TRACE_EVERY" "$SAMPLE_PERIOD_S" "$GPU_IDLE_THRESHOLD" "$NET_IFACE"
  if [[ "$NCCL_TRANSPORT_MODE" == tcp_observable ]]; then
    printf -v remote 'export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1; %s' "$remote"
  fi
  if [[ -n "${NCCL_IB_DISABLE:-}" ]]; then
    printf -v remote 'export NCCL_IB_DISABLE=%q; %s' "$NCCL_IB_DISABLE" "$remote"
  fi
  # Pass custom training args to every rank for compatibility/pilot use.
  if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then remote="$remote $EXTRA_TRAIN_ARGS"; fi
  (oarsh "$node" "$remote" > "$OUT/node_$i.log" 2>&1) &
  pids+=("$!")
done
if (( SHOW_PROGRESS )); then
  # Tail only rank-0 host, where training validation and summary print.
  tail -n 0 -F "$OUT/node_0.log" & tail_pid="$!"
fi
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
if [[ -n "$tail_pid" ]]; then kill "$tail_pid" 2>/dev/null || true;wait "$tail_pid" 2>/dev/null || true;tail_pid=''; fi
(( failed == 0 )) || die "Remote training failed"
if (( MEASURE_TCP )); then
  monitor_failed=0
  if [[ -s "$OUT/worker_map.json" ]]; then
    "$PYTHON" "$HERE/analyze_socket_counters.py" "$OUT" --windows-ms "${TRAFFIC_WINDOWS_MS:-100,1000}" || monitor_failed=1
    "$PYTHON" "$HERE/socket_monitor_health.py" "$OUT" --expected-hosts "$N_NODES" || monitor_failed=1
    "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 100 || true
    "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 1000 || true
  else
    monitor_failed=1
    echo 'WARNING: worker_map.json missing, no TCP attribution possible' >&2
  fi
  if (( monitor_failed )); then
    echo 'WARNING: TCP measurement incomplete; training data remains usable; missing traffic is not zero' >&2
  fi
fi
[[ -s "$OUT/summary.json" ]] || die "No summary.json produced"
log 'Create plots and aggregated metrics'
"$PYTHON" "$HERE/plot.py" "$OUT" --gpu-idle-threshold "$GPU_IDLE_THRESHOLD"
"$PYTHON" "$HERE/demand.py" "$OUT"
"$PYTHON" "$HERE/nccl_diagnostics.py" "$OUT"
"$PYTHON" "$HERE/communication_events.py" "$OUT"
"$PYTHON" "$HERE/plot_comparison.py" "$OUT"
"$PYTHON" "$HERE/comparison_metrics.py" "$OUT"
log 'E1_a completed'
echo "Results: $OUT"
echo "Plots: $OUT/plots"
