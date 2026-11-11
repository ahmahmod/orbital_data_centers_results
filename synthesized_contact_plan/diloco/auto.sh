#!/usr/bin/env bash
# E1_b DiLoCo outer-optimizer sweep: A1, A2, A3, B1, B2, C1.
# Run inside one OAR reservation containing the 4 GPU workers used by E1_b/run.sh.
# Uses the existing run.sh unchanged (including all metrics and plots).
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${MAX_SECONDS:=3600}"      # Per run; train.py also stops at epoch boundaries.
: "${EPOCHS:=100}"
: "${SEED:=2026}"
: "${N_NODES:=4}"
: "${GPUS_PER_NODE:=1}"
: "${DRY_RUN:=0}"           # DRY_RUN=1 prints commands, no GPUs required.
: "${FAIL_FAST:=1}"         # FAIL_FAST=0 continues even if a run fails.
: "${BATCH_ID:=diloco_outer_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${RESULTS_ROOT:=$HERE/results}"
: "${PYTHON:=python3}"

log() { printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$HERE/run.sh" ]] || die "Put this file in the E1_b folder next to run.sh"
[[ "$BATCH_ID" =~ ^[a-zA-Z0-9_.-]+$ ]] || die 'BATCH_ID must contain only letters, numbers, _, -, .'
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die 'DRY_RUN must be 0 or 1'
[[ "$FAIL_FAST" == 0 || "$FAIL_FAST" == 1 ]] || die 'FAIL_FAST must be 0 or 1'
[[ -x "$(command -v "$PYTHON" 2>/dev/null || true)" ]] || die "Cannot execute Python: $PYTHON"

# Prior H8 / LR1.0 / momentum0.9 is deliberately NOT repeated.
# Test ID / LOCAL_STEPS / OUTER_LR / OUTER_MOMENTUM
CONFIGS=(
    'A1 8 0.1 0.9'
    'A2 8 0.3 0.9'
    'A3 8 0.5 0.9'
    'B1 8 1.0 0.0'
    'B2 8 0.3 0.0'
    'C1 16 0.3 0.9'
)

if [[ "$DRY_RUN" == 0 ]]; then
    NODEFILE="${OAR_NODEFILE:-${OAR_FILE_NODES:-}}"
    [[ -n "$NODEFILE" && -r "$NODEFILE" ]] || die 'No active OAR reservation (OAR_NODEFILE missing). Reserve GPUs first.'
    mapfile -t NODE_LIST < <(sort -u "$NODEFILE")
    (( ${#NODE_LIST[@]} == N_NODES )) || die "Reservation has ${#NODE_LIST[@]} nodes; expected $N_NODES. Adjust N_NODES/GPUS_PER_NODE."
fi

BATCH_DIR="$RESULTS_ROOT/batches/$BATCH_ID"
INDEX="$BATCH_DIR/runs.tsv"
if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$BATCH_DIR" "$RESULTS_ROOT"
    if [[ ! -e "$INDEX" ]]; then
        printf 'test\tlocal_steps\touter_lr\touter_momentum\tstatus\trun_id\toutput\tbest_accuracy\tfinal_accuracy\twall_training_s\n' > "$INDEX"
    fi
fi

log "Executing DiLoCo sweep: ${#CONFIGS[@]} configurations, one after another"
log "Batch: $BATCH_ID | reservation: ${N_NODES} nodes x ${GPUS_PER_NODE} GPU(s)"
log "MAX_SECONDS=$MAX_SECONDS (per experiment), EPOCHS=$EPOCHS, SEED=$SEED"

failed=0
for line in "${CONFIGS[@]}"; do
    read -r test_id local_steps outer_lr outer_momentum <<< "$line"
    run_id="E1_b_diloco_${test_id}_H${local_steps}_OLR${outer_lr}_OM${outer_momentum}_${N_NODES}hosts_${GPUS_PER_NODE}gpu_seed${SEED}_${BATCH_ID}"
    out="$RESULTS_ROOT/$run_id"
    log "[$test_id] H=$local_steps, OUTER_LR=$outer_lr, OUTER_MOMENTUM=$outer_momentum"

    if [[ "$DRY_RUN" == 1 ]]; then
        printf 'METHOD=diloco LOCAL_STEPS=%q OUTER_LR=%q OUTER_MOMENTUM=%q MAX_SECONDS=%q EPOCHS=%q SEED=%q N_NODES=%q GPUS_PER_NODE=%q RUN_ID=%q OUT=%q ./run.sh\n' \
          "$local_steps" "$outer_lr" "$outer_momentum" "$MAX_SECONDS" "$EPOCHS" "$SEED" "$N_NODES" "$GPUS_PER_NODE" "$run_id" "$out"
        continue
    fi

    # Safe resumption: do not rerun completed configurations. Never overwrite a failed run.
    if [[ -s "$out/summary.json" ]] && "$PYTHON" - "$out/summary.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f: data=json.load(f)
sys.exit(0 if data.get('status')=='ok' else 1)
PY
    then
        log "[$test_id] Already completed: $out (skipping)"
        continue
    elif [[ -e "$out" ]]; then
        log "[$test_id] Existing incomplete output: $out; refusing to overwrite"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\n' "$test_id" "$local_steps" "$outer_lr" "$outer_momentum" 'incomplete_existing' "$run_id" "$out" >> "$INDEX"
        failed=$((failed+1))
        if [[ "$FAIL_FAST" == 1 ]]; then break; else continue; fi
    fi

    # run.sh collects logs, validation, GPU/TCP counters, and generates all plots.
    # All arguments are passed as exported environment vars, not string eval.
    status=success
    if METHOD=diloco LOCAL_STEPS="$local_steps" OUTER_LR="$outer_lr" \
       OUTER_MOMENTUM="$outer_momentum" MAX_SECONDS="$MAX_SECONDS" EPOCHS="$EPOCHS" \
       SEED="$SEED" N_NODES="$N_NODES" GPUS_PER_NODE="$GPUS_PER_NODE" \
       RUN_ID="$run_id" OUT="$out" bash "$HERE/run.sh"; then
        log "[$test_id] Training and plotting finished"
    else
        status=failed
        failed=$((failed+1))
        log "[$test_id] ERROR; inspect node logs under $out"
    fi

    best=NA; final=NA; wall=NA
    if [[ -s "$out/summary.json" ]]; then
        read -r best final wall < <("$PYTHON" - "$out/summary.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f: s=json.load(f)
last=s.get('latest_validation_rank0') or s.get('latest_validation') or {}
print(s.get('best_validation_accuracy_rank0',s.get('best_validation_accuracy','NA')),
      last.get('val_accuracy','NA'),s.get('wall_training_s','NA'))
PY
        )
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$test_id" "$local_steps" "$outer_lr" "$outer_momentum" "$status" "$run_id" "$out" "$best" "$final" "$wall" >> "$INDEX"

    if [[ "$status" == failed && "$FAIL_FAST" == 1 ]]; then break; fi
done

if [[ "$DRY_RUN" == 1 ]]; then
    log 'Dry run complete; no experiments launched.'
    exit 0
fi

log "Batch summary: $INDEX"
"$PYTHON" - "$INDEX" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1]); rows=list(csv.DictReader(p.open(),delimiter='\t'))
print(f'\n{"TEST":<6} {"H":>4} {"LR":>6} {"MOM":>6} {"STATUS":<20} {"BEST":>8} {"FINAL":>8} {"TIME":>10}')
for r in rows: print(f'{r["test"]:<6} {r["local_steps"]:>4} {r["outer_lr"]:>6} {r["outer_momentum"]:>6} {r["status"]:<20} {r["best_accuracy"]:>8} {r["final_accuracy"]:>8} {r["wall_training_s"]:>10}')
PY
if ((failed)); then
    log "$failed experiment(s) failed or already had incomplete outputs."
    exit 1
fi
log 'All requested DiLoCo configurations completed successfully.'
