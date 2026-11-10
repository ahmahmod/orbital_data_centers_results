#!/usr/bin/env bash
# oarsub -I -l "gpu=1,walltime=2:00:00"
# E0 single-GPU training. Run from an allocated GPU node, or from an OAR
# frontend that can reach its allocated nodes using oarsh.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
log() { printf '\n========== %s ==========\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

: "${MAX_SECONDS:=3600}"
: "${EPOCHS:=100}"
: "${SEED:=2026}"
: "${BATCH_SIZE:=32}"
: "${EVAL_EVERY:=200}"
: "${NUM_WORKERS:=4}"
: "${GPU_INDEX:=0}"
: "${DATA_ROOT:=$HERE/../data}"
: "${RUN_ID:=E0_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUT:=$HERE/results/$RUN_ID}"
# Set PYTHON=/path/to/python to override this preference.
: "${PYTHON:=$HERE/../.venv311/bin/python}"

[[ -f "$HERE/train.py" && -f "$HERE/plot.py" ]] || die "train.py and plot.py must be beside run.sh in $HERE"
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be an integer"

# The frontend generally has no NVIDIA GPU. In that case use the first OAR
# node, but do not launch a nested reservation or modify the reserved nodes.
if [[ "${E0_ON_COMPUTE:-0}" != 1 ]] && ! { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
    NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
    [[ -n "$NODEFILE" && -r "$NODEFILE" ]] || die "No GPU available and no OAR node file. Reserve a GPU first."
    NODE="$(sort -u "$NODEFILE" | head -n 1)"
    [[ -n "$NODE" ]] || die "OAR node file is empty"
    command -v oarsh >/dev/null 2>&1 || die "oarsh not found; run this script on the allocated GPU node"
    log "Executing E0 on reserved node: $NODE"
    printf -v REMOTE 'cd %q && E0_ON_COMPUTE=1 MAX_SECONDS=%q EPOCHS=%q SEED=%q BATCH_SIZE=%q EVAL_EVERY=%q NUM_WORKERS=%q DATA_ROOT=%q GPU_INDEX=%q RUN_ID=%q OUT=%q PYTHON=%q bash ./run.sh' \
      "$HERE" "$MAX_SECONDS" "$EPOCHS" "$SEED" "$BATCH_SIZE" "$EVAL_EVERY" "$NUM_WORKERS" "$DATA_ROOT" "$GPU_INDEX" "$RUN_ID" "$OUT" "$PYTHON"
    exec oarsh "$NODE" "$REMOTE"
fi

log 'Checking GPU and Python environment'
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi missing; run on a GPU node"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
if [[ ! -x "$PYTHON" ]]; then
    die "Python not found or not executable: $PYTHON. Create .venv311 here, or pass PYTHON=/actual/path/to/python"
fi
"$PYTHON" - <<'PY' || die "PyTorch, torchvision, CUDA, numpy, or matplotlib check failed"
import torch, torchvision, numpy, matplotlib
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('Using GPU:', torch.cuda.get_device_name(0))
PY

mkdir -p "$OUT" "$DATA_ROOT"
log "Training E0 on $(hostname)"
echo "Results: $OUT"
echo "Python:  $PYTHON"
echo "Duration: $MAX_SECONDS seconds"
# PIPESTATUS is preserved by pipefail; training failure will stop plotting.
"$PYTHON" "$HERE/train.py" --output "$OUT" --data-root "$DATA_ROOT" --download \
  --dataset cifar100 --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --seed "$SEED" --max-seconds "$MAX_SECONDS" \
  --target-accuracy 0.70 "$@" 2>&1 | tee -a "$OUT/console.log"

log 'Generating plots'
"$PYTHON" "$HERE/plot.py" "$OUT"
log 'Completed E0'
echo "Summary: $OUT/summary.json"
echo "Metrics: $OUT/steps.csv and $OUT/validation.csv"
echo "Plots:   $OUT/plots/"
