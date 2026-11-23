#!/usr/bin/env bash
# Run ON a deployed GPU node, preferably under the ordinary Grid'5000 user.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${BASE_PYTHON:-python3}"
if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  "$PY" -m venv "$HERE/.venv"
fi
VENV="$HERE/.venv/bin/python"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

"$VENV" -m pip install --upgrade pip
# CUDA 12.6. PyTorch wheels work with supported Blackwell builds and modern NVIDIA drivers.
"$VENV" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
"$VENV" -m pip install matplotlib numpy 'flwr==1.24.0'
"$VENV" - <<'PY'
import torch
import torchvision
import flwr
import numpy

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("CUDA wheel:", torch.version.cuda)

if torch.cuda.is_available():
    print("CUDA available: True")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA unavailable on this host. GPU verification skipped.")
PY
# Download CIFAR-100 if not already downloaded
if [[ ! -d "$HERE/data/cifar-100-python" ]]; then
    echo "Downloading CIFAR-100..."
    "$VENV" "$HERE/download_cifar.py"
else
    echo "CIFAR-100 already downloaded. Skipping."
fi
