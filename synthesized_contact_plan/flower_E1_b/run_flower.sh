#!/usr/bin/env bash
# Flower FedAvg add-on to E1_b. Launch from a Grid'5000 OAR GPU reservation.
# No OAR reservation, OS deployment, tc/netem, or download performed here.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
die(){ echo "ERROR: $*" >&2; exit 1; }
: "${N_NODES:=4}"; : "${GPUS_PER_NODE:=1}"
: "${ROUNDS:=100}"; : "${LOCAL_STEPS:=32}"; : "${MAX_SECONDS:=3600}"
: "${SEED:=2026}"; : "${BATCH_SIZE:=32}"; : "${NUM_WORKERS:=4}"
: "${DATA_ROOT:=$HERE/../data}"; : "${PYTHON:=$HERE/../.venv311/bin/python}"
: "${NET_IFACE:=br0}"; : "${PORT:=8085}"; : "${MEASURE_TCP:=1}"
: "${TCP_SAMPLE_PERIOD_S:=0.1}"; : "${SAMPLE_PERIOD_S:=1}"
: "${GPU_IDLE_THRESHOLD:=20}"; : "${RUN_ID:=E1_b_flower_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUT:=$HERE/results/$RUN_ID}"
[[ -x "$PYTHON" ]] || die "Missing Python at $PYTHON"
[[ -f "$HERE/e1a_reference.py" && -f "$HERE/plot.py" ]] || die 'Run ./install.sh first (copies E1_a_new reference/instrumentation)'
[[ -d "$DATA_ROOT/cifar-100-python" ]] || die "Dataset missing at $DATA_ROOT/cifar-100-python"
(( N_NODES * GPUS_PER_NODE == 4 )) || die 'Require four Flower clients total'
NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
[[ -n "$NODEFILE" && -r "$NODEFILE" ]] || die 'Run within an active OAR GPU reservation'
mapfile -t NODES < <(sort -u "$NODEFILE")
(( ${#NODES[@]} == N_NODES )) || die "Expected $N_NODES allocated hosts; found ${#NODES[@]}"
command -v oarsh >/dev/null || die 'oarsh is unavailable'
[[ ! -e "$OUT" ]] || die "Refusing to overwrite existing results: $OUT"
mkdir -p "$OUT"; OUT="$(realpath "$OUT")";DATA_ROOT="$(realpath "$DATA_ROOT")"

IPs=()
for node in "${NODES[@]}";do
  echo "Checking: $node"
  oarsh "$node" "'$PYTHON' -c 'import flwr,torch,torchvision,numpy,matplotlib;assert torch.cuda.is_available(); print(\"flower\",flwr.__version__,\"torch\",torch.__version__)'" || die "Flower/PyTorch/CUDA missing on $node; install Flower in shared venv"
  oarsh "$node" "test -d '$DATA_ROOT/cifar-100-python' && test -f '$HERE/train_flower.py' && test -w '$OUT'" || die "Shared paths not available on $node"
  count="$(oarsh "$node" 'nvidia-smi -L | wc -l' | tr -d '[:space:]')"
  ((count >= GPUS_PER_NODE)) || die "Not enough visible GPUs on $node ($count)"
  ip="$(oarsh "$node" "ip -4 -o addr show dev '$NET_IFACE' | awk '{split(\$4,a,\"/\"); print a[1];exit}'" | tr -d '\r')"
  [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "No IPv4 on $NET_IFACE for $node: $ip"
  IPs+=("$ip")
done
SERVER_ADDRESS="${IPs[0]}:${PORT}"
echo "Flower FedAvg E1_b: ${N_NODES} hosts x ${GPUS_PER_NODE} GPUs; $ROUNDS rounds; $LOCAL_STEPS local steps per round"
echo "Coordinator: $SERVER_ADDRESS | Results: $OUT"

pids=();server_pid='';tail_pid=''
cleanup(){ code=$?;trap - EXIT
  if [[ -n "$tail_pid" ]]; then kill "$tail_pid" 2>/dev/null || :;wait "$tail_pid" 2>/dev/null || :;fi
  if ((code != 0));then
    echo 'Flower run failed; inspect node/server logs:' >&2
    for file in "$OUT"/server.log "$OUT"/worker_*.log;do [[ -s "$file" ]] && tail -20 "$file" >&2;done
    for pid in "${pids[@]}";do kill "$pid" 2>/dev/null || :;done
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || :
  fi
}
trap cleanup EXIT

# The CPU-only Flower coordinator is hosted on NODE0; it does not consume a GPU.
printf -v server_command 'cd %q && %q %q --role server --server-address %q --world-size 4 --output %q --rounds %q --local-steps %q --batch-size %q --seed %q --target-accuracy 0.7 --max-seconds %q' \
  "$HERE" "$PYTHON" "$HERE/train_flower.py" "$SERVER_ADDRESS" "$OUT" "$ROUNDS" "$LOCAL_STEPS" "$BATCH_SIZE" "$SEED" "$MAX_SECONDS"
(oarsh "${NODES[0]}" "$server_command" > "$OUT/server.log" 2>&1) & server_pid=$!
sleep 3
if ! kill -0 "$server_pid" 2>/dev/null; then die 'Flower server exited before clients started';fi

rank=0
for ((i=0;i<N_NODES;i++));do
  for ((j=0;j<GPUS_PER_NODE;j++));do
    # CUDA_VISIBLE_DEVICES masks physical GPU j so each client always uses cuda:0.
    printf -v client_command 'cd %q && export CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=1 E1_WORKER_IP=%q E1_TCP_MEASURE=%q E1_TCP_PERIOD=%q E1_TCP_DURATION=%q && %q %q --role client --server-address %q --world-size 4 --rank %q --gpu-index %q --output %q --data-root %q --rounds %q --local-steps %q --epochs 100 --batch-size %q --num-workers %q --seed %q --max-seconds %q --sample-period-s %q --gpu-idle-threshold %q --net-iface %q' \
      "$HERE" "$j" "${IPs[$i]}" "$([[ "$j" == 0 ]] && echo "$MEASURE_TCP" || echo 0)" "$TCP_SAMPLE_PERIOD_S" "$((MAX_SECONDS+600))" \
      "$PYTHON" "$HERE/train_flower.py" "$SERVER_ADDRESS" "$rank" "$j" "$OUT" "$DATA_ROOT" "$ROUNDS" "$LOCAL_STEPS" "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "$MAX_SECONDS" "$SAMPLE_PERIOD_S" "$GPU_IDLE_THRESHOLD" "$NET_IFACE"
    (oarsh "${NODES[$i]}" "$client_command" > "$OUT/worker_$rank.log" 2>&1) & pids+=("$!")
    rank=$((rank+1))
  done
done

if [[ "${SHOW_PROGRESS:-1}" == 1 ]];then tail -n 0 -F "$OUT/server.log" &tail_pid=$!;fi
failure=0
for pid in "${pids[@]}";do wait "$pid" || failure=1;done
wait "$server_pid" || failure=1
if [[ -n "$tail_pid" ]];then kill "$tail_pid" 2>/dev/null || :;wait "$tail_pid" 2>/dev/null || :;tail_pid='';fi
((failure==0)) || die 'One or more Flower processes failed'
[[ -s "$OUT/summary.json" ]] || die 'No successful round summary generated'

# The original E1_b plot.py reads the identical per-rank validation and step schemas.
"$PYTHON" "$HERE/plot.py" "$OUT" --gpu-idle-threshold "$GPU_IDLE_THRESHOLD"
if [[ -f "$HERE/plot_flower.py" ]];then "$PYTHON" "$HERE/plot_flower.py" "$OUT";fi
if [[ "$MEASURE_TCP" == 1 && -f "$HERE/analyze_socket_counters.py" ]];then
  "$PYTHON" "$HERE/analyze_socket_counters.py" "$OUT" --windows-ms 100,1000 || echo 'WARNING: TCP analysis partial'
  "$PYTHON" "$HERE/socket_monitor_health.py" "$OUT" --expected-hosts "$N_NODES" || echo 'WARNING: TCP monitor incomplete'
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 100 || :
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 1000 || :
fi
echo "E1_b Flower run complete: $OUT"
