#!/usr/bin/env bash
# GPU-first E2 setup: deploy once, prepare network, install shared GPU Python environment.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export E2_REMOTE_MODE=deploy
# : "${ENVIRONMENT:=debian13-big}"
: "${NETWORK_MODE:=selective_tc}"
: "${DATASET:=synthetic}"
# : "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
[[ -n "${OAR_JOB_ID:-}" ]] || { echo 'ERROR: Run in a four-host OAR deploy reservation' >&2; exit 2; }

# ==========================================================
# Automatic GPU environment selection
# ==========================================================

select_gpu_environment() {
    local node cluster site
    local -a nodes

    mapfile -t nodes < <(sort -u "${OAR_NODEFILE:?}")

    if [[ "${#nodes[@]}" -ne 4 ]]; then
        echo "ERROR: Expected four allocated GPU nodes" >&2
        return 1
    fi

    node="${nodes[0]}"
    cluster="${node%%-*}"

    # Example: chifflot-3.lille.grid5000.fr -> lille
    site="${node#*.}"
    site="${site%%.*}"

    for node in "${nodes[@]}"; do
        if [[ "$node" != "${cluster}-"*".${site}.grid5000.fr" ]]; then
            echo "ERROR: Mixed clusters or sites: $node" >&2
            return 1
        fi
    done

    case "$site/$cluster" in
        lille/chifflot)
            AUTO_ENVIRONMENT="debiannvlegacy13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
        grenoble/kinovis)
            # NVIDIA L40S (Ada, SM 8.9) supports the open NVIDIA driver.
            AUTO_ENVIRONMENT="debian13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
        nancy/grue)
            # Tesla T4 (Turing, SM 7.5): supported by the standard Debian 13
            # GPU environment with NVIDIA open kernel modules.
            AUTO_ENVIRONMENT="debian13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
        nancy/granite)
            AUTO_ENVIRONMENT="debian13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu130"
            ;;
        lille/chuc)
            AUTO_ENVIRONMENT="debian13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
        *)
            # Conservative fallback for GPU clusters not explicitly profiled.
            # Runtime checks below still require NVIDIA GPU/CUDA on every host.
            echo "WARNING: No specific GPU environment for $site/$cluster; defaulting to debian13-big" >&2
            AUTO_ENVIRONMENT="debian13-big"
            AUTO_TORCH_INDEX="https://download.pytorch.org/whl/cu126"
            ;;
    esac

    echo "Detected GPU cluster: $site/$cluster"
}

select_gpu_environment

export ENVIRONMENT="${ENVIRONMENT:-$AUTO_ENVIRONMENT}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-$AUTO_TORCH_INDEX}"

echo "OS image: $ENVIRONMENT"
echo "PyTorch index: $TORCH_INDEX_URL"


[[ "${E2_CONFIRM_NETWORK:-0}" == 1 ]] || {
  echo 'Setup deploys/reuses four GPU nodes and configures networking.'
  echo 'Run: E2_CONFIRM_NETWORK=1 ./setup.sh'
  echo 'Default network mode is selective_tc; setup NEVER enables traffic shaping.'
  exit 2
}
command -v ssh >/dev/null
[[ -r "${OAR_NODEFILE:-/nonexistent}" ]] || { echo 'ERROR: OAR_NODEFILE missing' >&2; exit 2; }
mapfile -t NODES < <(sort -u "$OAR_NODEFILE")
[[ "${#NODES[@]}" == 4 ]] || { echo "ERROR: expected four hosts, got ${#NODES[@]}" >&2; exit 2; }
[[ "$DATASET" == synthetic || "$DATASET" == cifar100 ]] || { echo 'DATASET must be synthetic or cifar100' >&2; exit 2; }
STAMP="$HERE/.deployment_job"
if [[ "${E2_REDEPLOY:-0}" == 1 ]]; then
  echo "WARNING: forced OS redeployment on all 4 nodes: $ENVIRONMENT"
  python3 "$HERE/bootstrap.py" --deploy --apply --environment "$ENVIRONMENT" --network-mode "$NETWORK_MODE"
elif [[ -f "$STAMP" && "$(cat "$STAMP")" == "$OAR_JOB_ID" ]]; then
  echo "Reusing previously deployed OAR job $OAR_JOB_ID (no OS reinstall)"
  python3 "$HERE/bootstrap.py" --apply --network-mode "$NETWORK_MODE"
else
  echo "Deploying $ENVIRONMENT on OAR job $OAR_JOB_ID"
  python3 "$HERE/bootstrap.py" --deploy --apply --environment "$ENVIRONMENT" --network-mode "$NETWORK_MODE"
fi
# Do not stamp as healthy until network and GPU checks pass.
python3 "$HERE/verify_runtime.py"
printf '%s\n' "$OAR_JOB_ID" > "$STAMP"

# Installation MUST run with the deployed node's Python, not the frontend's.
# Use the ordinary account when available, since Grid'5000 shared /home is often root-squashed.
FIRST="${NODES[0]}"
LOGIN_USER="${E2_INSTALL_USER:-$(id -un)}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="$REPO_ROOT/.venv"
qhere="$(printf '%q' "$HERE")"
remote_py="$(printf '%q' "$VENV/bin/python")"

# Quick probe: correct interpreter and all runtime modules, including Flower.
probe="${remote_py} -c 'import sys,torch,torchvision,flwr,matplotlib; assert sys.version_info[:2] == (3,13), sys.version; assert torch.cuda.is_available(); assert sys.version_info[:2] == (3, 13); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'"
needs_install=0
if [[ "${E2_REINSTALL_DEPS:-0}" == 1 ]]; then
  needs_install=1
elif ! ssh -o BatchMode=yes "root@$FIRST" "$probe" >/dev/null 2>&1; then
  needs_install=1
fi
if ((needs_install)); then
  echo "Installing/repairing shared virtual environment on GPU host $FIRST"
  # Recreate only when explicitly requested; never silently delete an existing venv.
  if [[ "${E2_REBUILD_VENV:-0}" == 1 ]]; then
    echo 'Rebuilding .venv requested; removing previous environment.'
    # Try normal account on the shared filesystem; root may be NFS root-squashed.
    ssh -o BatchMode=yes "root@$FIRST" "runuser -u $(printf '%q' "$LOGIN_USER") -- rm -rf $(printf '%q' "$VENV")"
  fi
  # Quote remote values carefully, and run pip as the normal Grid'5000 user.
  ssh -o BatchMode=yes "root@$FIRST" \
    "id -u $(printf '%q' "$LOGIN_USER") >/dev/null && runuser -u $(printf '%q' "$LOGIN_USER") -- env TORCH_INDEX_URL=$(printf '%q' "$TORCH_INDEX_URL") BASE_PYTHON=python3 bash $(printf '%q' "$REPO_ROOT/shared_runtime.sh")" \
    || { echo 'ERROR: dependency installation failed; check GPU-node Python, write permissions and package logs.' >&2; exit 1; }
fi

# A shared path is not proof that the packages are usable on each GPU node.
for NODE in "${NODES[@]}"; do
  echo "CUDA/Flower check: $NODE"
  ssh -o BatchMode=yes "root@$NODE" "$probe" || {
    echo "ERROR: GPU Python verification failed on $NODE" >&2
    exit 1
  }
done

if [[ "$DATASET" == cifar100 && ! -d "$REPO_ROOT/data/cifar-100-python" ]]; then
  echo 'Downloading CIFAR-100 on the first GPU node as the ordinary user.'
  ssh -o BatchMode=yes "root@$FIRST" \
    "runuser -u $(printf '%q' "$LOGIN_USER") -- bash -c $(printf '%q' "cd '$HERE' && '$VENV/bin/python' download_cifar.py")"
fi

printf '\nSETUP PASS. OS=%s backend=%s python=%s\n' "$ENVIRONMENT" "$NETWORK_MODE" "$VENV/bin/python"
echo 'Dry run (no shaping): DATASET=synthetic METHOD=ddp MAX_SECONDS=180 ./run.sh'
echo 'WARNING: selective_tc replay changes the management NIC qdisc; do not enable shaping before separate safety checks.'
