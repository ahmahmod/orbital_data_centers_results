#!/usr/bin/env bash
# Run four workers on a *previously reserved* Grid'5000 OAR allocation.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
die(){ echo "ERROR: $*" >&2; exit 1; }
: "${METHOD:=diloco}"
: "${N_NODES:=4}"; : "${GPUS_PER_NODE:=1}"
: "${MAX_SECONDS:=3600}"; : "${EPOCHS:=100}"; : "${SEED:=2026}"
: "${BATCH_SIZE:=32}"; : "${NUM_WORKERS:=4}"; : "${LOG_EVERY_STEPS:=10}"
: "${DATA_ROOT:=$HERE/../data}"; : "${PYTHON:=$HERE/../.venv311/bin/python}"
: "${NET_IFACE:=br0}"; : "${MASTER_PORT:=29555}"
: "${NCCL_IB_DISABLE:=1}"; : "${SAMPLE_PERIOD_S:=1}"; : "${GPU_IDLE_THRESHOLD:=20}"
: "${MEASURE_TCP:=1}"; : "${TCP_SAMPLE_PERIOD_S:=0.1}"
: "${LOCAL_STEPS:=32}"; : "${OUTER_LR:=1.0}"; : "${OUTER_MOMENTUM:=0.9}"
: "${GOSSIP_EVERY:=32}"; : "${GOSSIP_ALPHA:=0.5}"
if [[ "$METHOD" == diloco ]]; then
  CONFIG_TAG="LOCAL_STEPS${LOCAL_STEPS}"
elif [[ "$METHOD" == gossip ]]; then
  CONFIG_TAG="GOSSIP_EVERY${GOSSIP_EVERY}"
else
  die 'METHOD must be diloco or gossip'
fi
: "${RUN_ID:=E1_b_${METHOD}_${CONFIG_TAG}_${N_NODES}hosts_${GPUS_PER_NODE}gpu_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUT:=$HERE/results/$RUN_ID}"
[[ "$METHOD" == diloco || "$METHOD" == gossip ]] || die 'METHOD must be diloco or gossip'
[[ -f "$HERE/e1a_reference.py" && -f "$HERE/plot.py" ]] || ./install.sh
[[ -x "$PYTHON" ]] || die "Missing Python $PYTHON"
[[ -d "$DATA_ROOT/cifar-100-python" ]] || die "Missing reusable CIFAR-100 at $DATA_ROOT/cifar-100-python"
NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
[[ -n "$NODEFILE" && -r "$NODEFILE" ]] || die 'Run inside an active OAR GPU reservation'
mapfile -t NODES < <(sort -u "$NODEFILE")
(( ${#NODES[@]} == N_NODES )) || die "Expected $N_NODES hosts; found ${#NODES[@]}: ${NODES[*]}"
command -v oarsh >/dev/null || die 'oarsh unavailable'
[[ ! -e "$OUT" ]] || die "Refusing to overwrite $OUT"
mkdir -p "$OUT";OUT="$(realpath "$OUT")";DATA_ROOT="$(realpath "$DATA_ROOT")"
HOST_IPS=()
for node in "${NODES[@]}"; do
  echo "Checking $node"
  oarsh "$node" "test -x '$PYTHON' && '$PYTHON' -c 'import torch,torchvision;assert torch.cuda.is_available();print(torch.__version__,torch.cuda.device_count())'" || die 'Torch/CUDA not available'
  gpu_count="$(oarsh "$node" 'nvidia-smi -L | wc -l' | tr -d '[:space:]')"
  (( gpu_count >= GPUS_PER_NODE )) || die "$node has $gpu_count GPUs, needs $GPUS_PER_NODE"
  oarsh "$node" "test -d '$DATA_ROOT/cifar-100-python' && test -f '$HERE/e1a_reference.py' && test -w '$OUT'" || die 'Shared storage missing'
  ip="$(oarsh "$node" "ip -4 -o addr show dev '$NET_IFACE' | awk '{split(\$4,a,\"/\"); print a[1];exit}'" | tr -d '\r')"
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "No IPv4 on $NET_IFACE for $node ($ip)"
  HOST_IPS+=("$ip")
done
MASTER_ADDR="${HOST_IPS[0]}"
echo "Running E1_b METHOD=$METHOD, ${N_NODES} hosts x ${GPUS_PER_NODE} GPUs, master ${MASTER_ADDR}:${MASTER_PORT}"
echo "Outputs: $OUT"
pids=();tail_pid=''
cleanup(){ code=$?; trap - EXIT; if [[ -n "$tail_pid" ]];then kill "$tail_pid" 2>/dev/null || :; wait "$tail_pid" 2>/dev/null || :;fi
  if (( code != 0 ));then
    echo "E1_b failed; see $OUT/node_*.log" >&2
    for f in "$OUT"/node_*.log;do [[ -f "$f" ]] && tail -25 "$f" >&2;done
    for p in "${pids[@]}";do kill "$p" 2>/dev/null || :;done
  fi; }
trap cleanup EXIT
for ((i=0;i<N_NODES;i++));do
  printf -v command 'cd %q && export OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=%q GLOO_SOCKET_IFNAME=%q NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=%q E1_WORKER_IP=%q E1_TCP_MEASURE=%q E1_TCP_PERIOD=%q E1_TCP_DURATION=%q && %q -m torch.distributed.run --nnodes=%q --nproc-per-node=%q --node-rank=%q --master-addr=%q --master-port=%q %q --output=%q --data-root=%q --method=%q --epochs=%q --batch-size=%q --num-workers=%q --seed=%q --max-seconds=%q --log-every-steps=%q --sample-period-s=%q --gpu-idle-threshold=%q --net-iface=%q --local-steps=%q --outer-lr=%q --outer-momentum=%q --gossip-every=%q --gossip-alpha=%q' \
    "$HERE" "$NET_IFACE" "$NET_IFACE" "$NCCL_IB_DISABLE" "${HOST_IPS[$i]}" "$MEASURE_TCP" "$TCP_SAMPLE_PERIOD_S" "$((MAX_SECONDS+180))" \
    "$PYTHON" "$N_NODES" "$GPUS_PER_NODE" "$i" "$MASTER_ADDR" "$MASTER_PORT" "$HERE/train.py" \
    "$OUT" "$DATA_ROOT" "$METHOD" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "$MAX_SECONDS" "$LOG_EVERY_STEPS" "$SAMPLE_PERIOD_S" "$GPU_IDLE_THRESHOLD" "$NET_IFACE" \
    "$LOCAL_STEPS" "$OUTER_LR" "$OUTER_MOMENTUM" "$GOSSIP_EVERY" "$GOSSIP_ALPHA"
  if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]];then command="$command $EXTRA_TRAIN_ARGS";fi
  (oarsh "${NODES[$i]}" "$command" >"$OUT/node_$i.log" 2>&1) & pids+=("$!")
done
if [[ "${SHOW_PROGRESS:-1}" == 1 ]];then tail -n 0 -F "$OUT/node_0.log" & tail_pid="$!";fi
failed=0
for p in "${pids[@]}";do wait "$p" || failed=1;done
if [[ -n "$tail_pid" ]];then kill "$tail_pid" 2>/dev/null || :;wait "$tail_pid" 2>/dev/null || :;tail_pid='';fi
(( failed == 0 )) || die 'At least one worker failed'
if [[ "$MEASURE_TCP" == 1 ]];then
  "$PYTHON" "$HERE/analyze_socket_counters.py" "$OUT" --windows-ms 100,1000 || echo 'WARNING: TCP samples incomplete'
  "$PYTHON" "$HERE/socket_monitor_health.py" "$OUT" --expected-hosts "$N_NODES" || echo 'WARNING: TCP health incomplete'
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 100 || :
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 1000 || :
fi
[[ -s "$OUT/summary.json" ]] || die 'Missing training summary'
"$PYTHON" "$HERE/plot.py" "$OUT" --gpu-idle-threshold "$GPU_IDLE_THRESHOLD"
"$PYTHON" "$HERE/demand_b.py" "$OUT"
if [[ "$METHOD" == gossip ]]; then
  "$PYTHON" "$HERE/plot_gossip_validation.py" "$OUT"
fi
echo "Completed E1_b: $OUT"
