#!/usr/bin/env bash
# Reserve four entire GPU hosts on ONE Grid'5000 site plus one local KaVLAN.
# SITE=lille|grenoble|nancy ./reservation.sh
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="${SITE:-$(hostname -s | sed -nE 's/^f(nancy|lille|grenoble)$/\1/p')}"
[[ -n "$SITE" ]] || { echo 'Set SITE=lille|grenoble|nancy and run on its frontend' >&2; exit 2; }
if [[ -n "${OAR_JOB_ID:-}" ]]; then
  echo "Existing OAR job: $OAR_JOB_ID"; sort -u "${OAR_NODEFILE:?}"; echo "Do not reserve twice; use E2_CONFIRM_NETWORK=1 ./setup.sh to deploy and configure."; exit 0
fi
[[ "$(hostname -s)" == "f$SITE" ]] || { echo "Please run on f$SITE (current: $(hostname -s))" >&2; exit 2; }
WALLTIME="${WALLTIME:-4:00:00}"
CLUSTER="${CLUSTER:-}"
QUEUE="${QUEUE:-}"
if [[ -z "$CLUSTER" ]]; then
  case "$SITE" in lille)CLUSTER=chifflot;; grenoble)CLUSTER=kinovis;; nancy)CLUSTER=gruss;; *)echo 'Unknown site' >&2;exit 2;;esac
fi
if [[ -z "$QUEUE" ]]; then
  if [[ "$SITE/$CLUSTER" == nancy/grue ]]; then
    QUEUE=production
  else
    QUEUE=abaca
  fi
fi
# The cluster predicate is deliberately attached to the 4 HOST resource set.
# Queue/site constraints may change; fail rather than silently booking CPU hosts.
MODE="deploy"
[[ "${MODE:-deploy}" == deploy ]] || { echo "E2 requires MODE=deploy for KaVLAN network reconfiguration" >&2; exit 2; }
TYPES=()
[[ "$MODE" == deploy ]] && TYPES=(-t deploy)
RESOURCE="{type='kavlan-local'}/vlan=1 + {cluster='$CLUSTER' AND gpu_count > 0}/host=4,walltime=$WALLTIME"
echo "SITE=$SITE CLUSTER=$CLUSTER QUEUE=$QUEUE WALLTIME=$WALLTIME" >&2
echo "oarsub -I -q $QUEUE ${TYPES[*]} -l \"$RESOURCE\"" >&2
exec oarsub -I -q "$QUEUE" "${TYPES[@]}" -l "$RESOURCE"
