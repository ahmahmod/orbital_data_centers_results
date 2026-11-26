#!/usr/bin/env bash
# E3_1 paired contact-aware benchmark. Call from technique-specific run.sh.
# DDP and DiLoCo support CASE=off|aware|both.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:?Set METHOD to ddp, diloco8, diloco64, gossip8, gossip64, fl8, or fl64}"
CASE="${CASE:-$( [[ "$METHOD" == ddp ]] && echo aware || echo both )}"
case "$METHOD" in
  ddp|diloco8|diloco64|gossip8|gossip64|fl8|fl64) ;;
  *) echo "Unknown METHOD=$METHOD" >&2;exit 2;;
esac
case "$CASE" in
  both) MODES=(off aware) ;;
  off|aware) MODES=("$CASE") ;;
  *) echo 'CASE=both|off|aware' >&2; exit 2 ;;
esac
: "${TC_APPLY:=1}"
: "${E2_CONFIRM_SHAPING:=1}"
: "${MAX_SECONDS:=10800}"
: "${DATASET:=cifar100}"
: "${E3_MARGIN_S:=15}"
: "${E3_ESTIMATE_FACTOR:=2}"
export TC_APPLY E2_CONFIRM_SHAPING MAX_SECONDS DATASET E3_MARGIN_S E3_ESTIMATE_FACTOR

# Reserve enough walltime for sequential trials, including setup and shutdown.
# Running both cannot fit a typical 4-hour reservation at MAX_SECONDS=10800.
if [[ "$CASE" == both && "$MAX_SECONDS" -ge 3600 && "${E3_CONFIRM_LONG_PAIR:-0}" != 1 ]];then
  echo "Both runs require roughly twice MAX_SECONDS plus setup/grace." >&2
  echo "Use separate reservations (CASE=off then CASE=aware) or set E3_CONFIRM_LONG_PAIR=1." >&2
  exit 2
fi
if [[ -z "${OAR_JOB_ID:-}" ]];then
 echo "Run inside an OAR GPU reservation. Start from $HERE/reservation.sh" >&2;exit 2
fi
case "$METHOD" in
  ddp) TECH_DIR="$HERE/ddp" ;;
  diloco8|diloco64) TECH_DIR="$HERE/diloco" ;;
  gossip8|gossip64) TECH_DIR="$HERE/gossip" ;;
  fl8|fl64) TECH_DIR="$HERE/flower" ;;
esac
PAIR_ID="${E3_PAIR_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}}"
FAIL=0
for mode in "${MODES[@]}";do
  echo "======= E3_1 METHOD=$METHOD SCHEDULE=$mode MAX_SECONDS=$MAX_SECONDS ======="
  OUTPUT="$TECH_DIR/results/E3_1_${METHOD}_${mode}_${PAIR_ID}"
  if E3_SCHEDULE_MODE="$mode" "$HERE/run.sh" --out "$OUTPUT" "$@";then
     echo "FINISHED: $OUTPUT"
  else
     status=$?
     echo "WARNING: $mode exited $status; preserving raw logs for comparison." >&2
     FAIL=1
  fi
done
if [[ "$CASE" == both ]];then
  TARGET_MODE=aware
  OFF="$TECH_DIR/results/E3_1_${METHOD}_off_${PAIR_ID}"
  ON="$TECH_DIR/results/E3_1_${METHOD}_${TARGET_MODE}_${PAIR_ID}"
  if [[ -f "$OFF/e3_summary.json" && -f "$ON/e3_summary.json" ]];then
     python3 "$HERE/compare_pair.py" --off "$OFF" --aware "$ON" \
       --out "$TECH_DIR/comparisons/${METHOD}_off_vs_${TARGET_MODE}_${PAIR_ID}" || FAIL=1
  else
     echo 'Pair comparison requires e3_summary.json from BOTH runs.' >&2
     FAIL=1
  fi
fi
exit "$FAIL"
