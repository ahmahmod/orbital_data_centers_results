#!/usr/bin/env bash
# E1_b Flower FedAvg: robust Grid'5000 four-GPU launcher.
# REPLACES flower_E1_b/run_flower.sh; train_flower.py is unchanged.
# Requires an existing OAR allocation and shared paths on all allocated hosts.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
error() { printf 'ERROR: %s\n' "$*" >&2; }
die() { error "$*"; exit 1; }

: "${N_NODES:=4}"; : "${GPUS_PER_NODE:=1}"
: "${ROUNDS:=500}"; : "${LOCAL_STEPS:=8}"; : "${MAX_SECONDS:=3600}"
: "${SEED:=2026}"; : "${BATCH_SIZE:=32}"; : "${NUM_WORKERS:=0}"
: "${DATA_ROOT:=$HERE/../data}"; : "${PYTHON:=$HERE/../.venv311/bin/python}"
: "${NET_IFACE:=br0}"; : "${PORT:=8085}"; : "${MEASURE_TCP:=1}"
: "${TCP_SAMPLE_PERIOD_S:=0.1}"; : "${SAMPLE_PERIOD_S:=1}"
: "${GPU_IDLE_THRESHOLD:=20}"; : "${SERVER_START_TIMEOUT:=90}"
: "${CHECK_PERIOD:=2}"; : "${SHOW_PROGRESS:=1}"
: "${RUN_ID:=E1_b_flower_H${LOCAL_STEPS}_R${ROUNDS}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUT:=$HERE/results/$RUN_ID}"

[[ -x "$PYTHON" ]] || die "Missing Python: $PYTHON"
[[ -f "$HERE/train_flower.py" ]] || die 'Missing train_flower.py'
[[ -f "$HERE/e1a_reference.py" && -f "$HERE/plot.py" ]] || die 'Run ./install.sh first'
[[ -d "$DATA_ROOT/cifar-100-python" ]] || die "Dataset missing: $DATA_ROOT/cifar-100-python"
(( N_NODES * GPUS_PER_NODE == 4 )) || die 'Require exactly four Flower clients'
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || die 'PORT must be 1024..65535'
NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
[[ -n "$NODEFILE" && -r "$NODEFILE" ]] || die 'Run inside an active OAR GPU reservation'
mapfile -t NODES < <(sort -u "$NODEFILE")
(( ${#NODES[@]} == N_NODES )) || die "Expected $N_NODES hosts; found ${#NODES[@]}"
command -v oarsh >/dev/null || die 'oarsh is unavailable'
[[ ! -e "$OUT" ]] || die "Refusing to overwrite existing results: $OUT"
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
DATA_ROOT="$(realpath "$DATA_ROOT")"

# A small socket check using system Python, without importing the ML stack.
port_open() {
  python3 - "$1" "$2" <<'PYSOCKET'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2.0):
        pass
except OSError:
    sys.exit(1)
PYSOCKET
}

# Verify the remote Flower Python PID, not just the local `oarsh` wrapper.
remote_process_alive() {
  local node="$1" file="$2" remote_pid
  [[ -s "$file" ]] || return 1
  remote_pid="$(cat "$file" 2>/dev/null || :)"
  [[ "$remote_pid" =~ ^[0-9]+$ ]] || return 1
  local qout
  printf -v qout '%q' "$OUT"
  oarsh "$node" "test -r /proc/$remote_pid/cmdline && tr '\\0' ' ' < /proc/$remote_pid/cmdline | grep -F 'train_flower.py' | grep -F -- $qout >/dev/null" >/dev/null 2>&1
}

# Distinguish the coordinator's own reachability from launcher reachability.
server_node_port_open() {
  local node="$1" host="$2" port="$3"
  oarsh "$node" "python3 -c 'import socket; s=socket.create_connection((\"$host\", $port), timeout=2); s.close()'" >/dev/null 2>&1
}

IPs=()
for node in "${NODES[@]}"; do
    echo "Checking: $node"
    oarsh "$node" "'$PYTHON' -c 'import flwr,torch,torchvision,numpy,matplotlib; assert torch.cuda.is_available(); print(\"flower\",flwr.__version__,\"torch\",torch.__version__)'" \
      || die "Flower/PyTorch/CUDA missing on $node"
    oarsh "$node" "test -d '$DATA_ROOT/cifar-100-python' && test -f '$HERE/train_flower.py' && test -w '$OUT'" \
      || die "Shared files not available on $node"
    gpu_count="$(oarsh "$node" 'nvidia-smi -L | wc -l' | tr -d '[:space:]')"
    (( gpu_count >= GPUS_PER_NODE )) || die "$node has $gpu_count GPUs; needs $GPUS_PER_NODE"
    ip="$(oarsh "$node" "ip -4 -o addr show dev '$NET_IFACE' | awk '{split(\$4,a,\"/\"); print a[1];exit}'" | tr -d '\r')"
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "No IPv4 on $NET_IFACE for $node"
    IPs+=("$ip")
done
SERVER_HOST="${IPs[0]}"
SERVER_ADDRESS="${SERVER_HOST}:${PORT}"
if port_open "$SERVER_HOST" "$PORT"; then
    die "TCP port $SERVER_ADDRESS is already open. Stop the old server or set PORT=8086."
fi

echo "Flower FedAvg: $N_NODES hosts x $GPUS_PER_NODE GPUs, $ROUNDS rounds, $LOCAL_STEPS steps/round"
echo "Coordinator: $SERVER_ADDRESS"
echo "Output: $OUT"
echo "OAR_JOB_ID=${OAR_JOB_ID:-unknown}"
printf 'run_id\t%s\noar_job_id\t%s\nserver_address\t%s\n' "$RUN_ID" "${OAR_JOB_ID:-unknown}" "$SERVER_ADDRESS" > "$OUT/launcher_metadata.tsv"

server_pid=''; tail_pid=''; pids=(); types=(); ranks=(); nodes=(); pidfiles=()
server_completed=0
cleaning=0
show_logs() {
    echo '================ SERVER LOG ================' >&2
    [[ -f "$OUT/server.log" ]] && tail -n 70 "$OUT/server.log" >&2 || :
    for rank in 0 1 2 3; do
        echo "================ CLIENT $rank ================" >&2
        [[ -f "$OUT/worker_$rank.log" ]] && tail -n 45 "$OUT/worker_$rank.log" >&2 || echo 'No worker log' >&2
    done
    echo '================ PROCESS EXIT RECORD ================' >&2
    [[ -f "$OUT/process_exits.tsv" ]] && cat "$OUT/process_exits.tsv" >&2 || :
    echo '================ SERVER STATE ================' >&2
    [[ -f "$OUT/summary.json" ]] && cat "$OUT/summary.json" >&2 || :
}

# Kill ONLY processes started by this launcher. PID files are written by the
# remote shell immediately before exec'ing train_flower.py. Check cmdline so
# an unrelated process with a reused PID cannot be terminated accidentally.
stop_remote() {
    local node="$1" file="$2"
    [[ -f "$file" ]] || return 0
    local pid
    pid="$(cat "$file" 2>/dev/null || :)"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    local qout
    printf -v qout '%q' "$OUT"
    oarsh "$node" "pid=$pid; if test -r /proc/\$pid/cmdline && tr '\\0' ' ' </proc/\$pid/cmdline | grep -F 'train_flower.py' | grep -F -- $qout >/dev/null; then kill -TERM \$pid 2>/dev/null || :; fi" \
        >/dev/null 2>&1 || :
}

cleanup() {
    local rc=$?
    (( cleaning == 0 )) || return
    cleaning=1
    trap - EXIT INT TERM HUP
    if [[ -n "$tail_pid" ]]; then kill "$tail_pid" 2>/dev/null || :; wait "$tail_pid" 2>/dev/null || :; fi
    if (( rc != 0 )); then
        error "Flower run failed (exit code $rc); logs in $OUT"
        show_logs
        # Stop any processes still active on the reservation; don't rely on
        # killing oarsh wrappers alone (the remote Python may survive).
        for ((i=0; i<${#pidfiles[@]}; i++)); do
            stop_remote "${nodes[$i]}" "${pidfiles[$i]}"
        done
        for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || :; done
        [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || :
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'error "Received SIGINT"; exit 130' INT
trap 'error "Received SIGTERM (possible OAR preemption or external kill)"; exit 143' TERM
trap 'error "Received SIGHUP"; exit 129' HUP

# PID belongs to the remote command shell and remains the same after exec.
# Quoting through printf %q prevents spaces or shell metacharacters in paths.
server_file="$OUT/server_remote.pid"
printf -v server_command 'cd %q && echo $$ > %q && exec %q -u %q --role server --server-address %q --world-size 4 --output %q --rounds %q --local-steps %q --batch-size %q --seed %q --target-accuracy 0.7 --max-seconds %q' \
  "$HERE" "$server_file" "$PYTHON" "$HERE/train_flower.py" "$SERVER_ADDRESS" "$OUT" "$ROUNDS" "$LOCAL_STEPS" "$BATCH_SIZE" "$SEED" "$MAX_SECONDS"
(oarsh "${NODES[0]}" "$server_command" > "$OUT/server.log" 2>&1) & server_pid=$!
pids+=("$server_pid");types+=(server);ranks+=(-1);nodes+=("${NODES[0]}");pidfiles+=("$server_file")
printf 'kind\trank\thost\tlocal_oarsh_pid\tremote_pid_file\nserver\t-1\t%s\t%s\t%s\n' "${NODES[0]}" "$server_pid" "$server_file" > "$OUT/processes.tsv"
printf 'timestamp_utc\tkind\trank\tlocal_pid\texit_code\n' > "$OUT/process_exits.tsv"

echo "Waiting for Flower server TCP readiness (up to ${SERVER_START_TIMEOUT}s)..."
printf 'elapsed_s\tlauncher_tcp\tserver_node_tcp\tremote_process\toarsh_wrapper\n' > "$OUT/server_readiness.tsv"
server_ready=0
for ((sec=0; sec<SERVER_START_TIMEOUT; sec++)); do
    # TCP is the authoritative readiness signal. Do not block on wait here.
    if port_open "$SERVER_HOST" "$PORT"; then
        printf '%s\t1\t-\t-\t-\n' "$sec" >> "$OUT/server_readiness.tsv"
        server_ready=1
        break
    fi
    node_tcp=0; remote_alive=0; wrapper_alive=0
    if server_node_port_open "${NODES[0]}" "$SERVER_HOST" "$PORT"; then node_tcp=1; fi
    if remote_process_alive "${NODES[0]}" "$server_file"; then remote_alive=1; fi
    if kill -0 "$server_pid" 2>/dev/null; then wrapper_alive=1; fi
    printf '%s\t0\t%s\t%s\t%s\n' "$sec" "$node_tcp" "$remote_alive" "$wrapper_alive" >> "$OUT/server_readiness.tsv"
    if (( sec == 0 || sec % 5 == 0 )); then
        echo "Readiness ${sec}s: launcher_tcp=0 server_node_tcp=$node_tcp remote_python=$remote_alive oarsh=$wrapper_alive"
    fi
    if (( remote_alive == 0 && wrapper_alive == 0 && sec >= 5 )); then
        tail -n 35 "$OUT/server.log" >&2 || :
        die 'Flower server and its oarsh wrapper both exited before readiness'
    fi
    sleep 1
done
if (( server_ready != 1 )); then
    echo '===== LAST READINESS PROBES =====' >&2
    tail -n 20 "$OUT/server_readiness.tsv" >&2 || :
    echo '===== COORDINATOR SOCKETS =====' >&2
    oarsh "${NODES[0]}" "ss -ltnp '( sport = :$PORT )' || :" >&2 || :
    die "Server not reachable from launcher at $SERVER_ADDRESS within ${SERVER_START_TIMEOUT}s"
fi
echo 'Server TCP listener is ready; starting all four clients.'

rank=0
for ((i=0; i<N_NODES; i++)); do
  for ((j=0; j<GPUS_PER_NODE; j++)); do
    client_file="$OUT/worker_${rank}_remote.pid"
    tcp_measure=0
    (( j != 0 )) || tcp_measure="$MEASURE_TCP"
    printf -v client_command 'cd %q && echo $$ > %q && export CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=1 E1_WORKER_IP=%q E1_TCP_MEASURE=%q E1_TCP_PERIOD=%q E1_TCP_DURATION=%q && exec %q -u %q --role client --server-address %q --world-size 4 --rank %q --gpu-index %q --output %q --data-root %q --rounds %q --local-steps %q --epochs 100 --batch-size %q --num-workers %q --seed %q --max-seconds %q --sample-period-s %q --gpu-idle-threshold %q --net-iface %q' \
      "$HERE" "$client_file" "$j" "${IPs[$i]}" "$tcp_measure" "$TCP_SAMPLE_PERIOD_S" "$((MAX_SECONDS+600))" \
      "$PYTHON" "$HERE/train_flower.py" "$SERVER_ADDRESS" "$rank" "$j" "$OUT" "$DATA_ROOT" "$ROUNDS" "$LOCAL_STEPS" "$BATCH_SIZE" "$NUM_WORKERS" "$SEED" "$MAX_SECONDS" "$SAMPLE_PERIOD_S" "$GPU_IDLE_THRESHOLD" "$NET_IFACE"
    (oarsh "${NODES[$i]}" "$client_command" > "$OUT/worker_${rank}.log" 2>&1) & pid=$!
    pids+=("$pid"); types+=(client);ranks+=("$rank");nodes+=("${NODES[$i]}");pidfiles+=("$client_file")
    printf 'client\t%s\t%s\t%s\t%s\n' "$rank" "${NODES[$i]}" "$pid" "$client_file" >> "$OUT/processes.tsv"
    echo "Started client rank $rank on ${NODES[$i]} (oarsh PID $pid)"
    rank=$((rank+1))
  done
done


# Build mapping required by TCP socket attribution.
echo "Waiting for Flower client PID files..."

for rank in 0 1 2 3; do
    pidfile="$OUT/worker_${rank}_remote.pid"
    ready=0
    for attempt in {1..30}; do
        if [[ -s "$pidfile" ]]; then
            ready=1
            break
        fi
        sleep 1
    done
    (( ready == 1 )) || die "Missing remote PID for client $rank"
done

"$PYTHON" - "$OUT" "${NODES[@]}" -- "${IPs[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
separator = sys.argv.index("--")
nodes = sys.argv[2:separator]
ips = sys.argv[separator + 1:]

workers = []
for rank, (node, ip) in enumerate(zip(nodes, ips)):
    pid = int((out / f"worker_{rank}_remote.pid").read_text().strip())
    workers.append({
        "rank": rank,
        "node": node,
        "ip": ip,
        "pid": pid,
    })

temporary = out / "worker_map.json.tmp"
temporary.write_text(json.dumps({"workers": workers}, indent=2))
temporary.replace(out / "worker_map.json")

print("Worker mapping created:", workers)
PY

if [[ "$SHOW_PROGRESS" == 1 ]]; then
    tail -n 0 -F "$OUT/server.log" & tail_pid=$!
fi

# Monitor all launch processes, even those that exit before this loop begins.
# Unlike wait -n, explicit wait PID can retrieve already-completed child statuses.
active=("${pids[@]}")
while (( ${#active[@]} > 0 )); do
    next=()
    for pid in "${active[@]}"; do
        # `jobs -pr` excludes exited jobs, unlike `kill -0` on zombies.
        if jobs -pr | grep -Fx -- "$pid" >/dev/null; then
            next+=("$pid")
            continue
        fi
        if wait "$pid"; then rc=0; else rc=$?; fi
        index=-1
        for ((k=0;k<${#pids[@]};k++)); do
            if [[ "${pids[$k]}" == "$pid" ]]; then index=$k;break;fi
        done
        (( index >= 0 )) || die "Unknown completed PID $pid"
        printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${types[$index]}" "${ranks[$index]}" "$pid" "$rc" >> "$OUT/process_exits.tsv"
        echo "Process exited: ${types[$index]} rank=${ranks[$index]} host=${nodes[$index]} code=$rc"
        (( rc == 0 )) || die "${types[$index]} rank=${ranks[$index]} failed with exit code $rc"
        if [[ "${types[$index]}" == server ]]; then
            server_completed=1
            "$PYTHON" - "$OUT/summary.json" "$ROUNDS" <<'PY_CHECK' || die 'Server exited before all requested FedAvg rounds completed'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): sys.exit(1)
d=json.loads(p.read_text())
if d.get('status') != 'ok' or int(d.get('completed_rounds',-1)) != int(sys.argv[2]): sys.exit(1)
PY_CHECK
        fi
    done
    active=("${next[@]}")
    (( ${#active[@]} == 0 )) || sleep "$CHECK_PERIOD"
done
(( server_completed == 1 )) || die 'Server completion not observed'

# The original E1_b plot.py reads the same training/validation CSV schemas.
"$PYTHON" "$HERE/plot.py" "$OUT" --gpu-idle-threshold "$GPU_IDLE_THRESHOLD"
if [[ -f "$HERE/plot_flower.py" ]]; then "$PYTHON" "$HERE/plot_flower.py" "$OUT"; fi
if [[ "$MEASURE_TCP" == 1 && -f "$HERE/analyze_socket_counters.py" ]]; then
  "$PYTHON" "$HERE/analyze_socket_counters.py" "$OUT" --windows-ms 100,1000 || echo 'WARNING: TCP analysis incomplete'
  "$PYTHON" "$HERE/socket_monitor_health.py" "$OUT" --expected-hosts "$N_NODES" || echo 'WARNING: TCP health incomplete'
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 100 || :
  "$PYTHON" "$HERE/plot_tcp_counters.py" "$OUT" --window-ms 1000 || :
fi
echo "Flower FedAvg run complete: $OUT"
