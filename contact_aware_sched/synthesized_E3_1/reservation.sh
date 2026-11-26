#!/usr/bin/env bash
# Request exact Nancy Graffiti GPU nodes in one deploy+KaVLAN-local reservation.
# Grid'5000 OAR predicates may differ between site/queue versions; fail fast.
set -Eeuo pipefail
EXPECTED=(graffiti-5 graffiti-7 graffiti-8 graffiti-9)
if [[ -n "${OAR_JOB_ID:-}" ]];then
  mapfile -t ALLOC < <(sort -u "${OAR_NODEFILE:?}" | sed 's/\..*//' | sort)
  [[ "${ALLOC[*]}" == "${EXPECTED[*]}" ]] || {
     printf 'Expected graffiti-5,7,8,9; allocated: %s\n' "${ALLOC[*]}" >&2;exit 2;
  }
  echo "Existing E3 allocation: ${ALLOC[*]}";exit 0
fi
[[ "$(hostname -s)" == fnancy ]] || { echo 'Run on fnancy Grid5000 frontend' >&2;exit 2; }
QUEUE="${QUEUE:-abaca}"
WALLTIME="${WALLTIME:-4:00:00}"
FILTER="host IN ('graffiti-5.nancy.grid5000.fr','graffiti-7.nancy.grid5000.fr','graffiti-8.nancy.grid5000.fr','graffiti-9.nancy.grid5000.fr')"
RESOURCE="{type='kavlan-local'}/vlan=1 + {${FILTER} AND gpu_count > 0}/host=4,walltime=${WALLTIME}"
echo "Requesting exact Graffiti nodes, queue $QUEUE, walltime $WALLTIME"
echo "oarsub -I -q $QUEUE -t deploy -l \"$RESOURCE\""
exec oarsub -I -q "$QUEUE" -t deploy -l "$RESOURCE"
