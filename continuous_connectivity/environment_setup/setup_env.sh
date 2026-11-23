#!/usr/bin/env bash
# Build a reproducible Python 3.11 CUDA 11.8 environment for Grid'5000 E0.
# Run in the same directory as run.sh, on shared persistent storage.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export PATH="$HOME/.local/bin:$PATH"

# Step 1 — Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv automatically..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version:"
uv --version

# Step 2 — Install Python 3.11
echo "Installing Python 3.11..."
uv python install 3.11

# Step 3 — Create the virtual environment
echo "Setting up virtual environment..."
if [[ ! -x .venv311/bin/python ]]; then
  uv venv --python 3.11 .venv311
fi

# Verify Python version inside the environment
.venv311/bin/python - <<'PY'
import sys
assert sys.version_info[:2] == (3,11), f'Expected Python 3.11, got {sys.version}'
print(f"Environment Python verified: {sys.version.split()[0]}")
PY

# Step 4 — Install PyTorch for Tesla P100 and dependencies
echo "Installing PyTorch 2.5.1 (CUDA 11.8) and extra dependencies..."
uv pip install --python .venv311/bin/python \
  torch==2.5.1 \
  torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu118

uv pip install --python .venv311/bin/python \
  numpy \
  matplotlib \
  pillow

# Optional: If you still have a requirements.txt file for other dependencies
if [[ -f requirements.txt ]]; then
  echo "Installing additional packages from requirements.txt..."
  uv pip install --python .venv311/bin/python -r requirements.txt
fi

# # Step 5 — Verify the GPU
# echo "Running GPU validation tests..."
# .venv311/bin/python - <<'PY'
# import sys
# import torch
# import torchvision

# print("PyTorch:", torch.__version__)
# print("Torchvision:", torchvision.__version__)
# print("CUDA runtime:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())

# assert torch.cuda.is_available(), "GPU not available"

# print("GPU:", torch.cuda.get_device_name(0))
# print("Capability:", torch.cuda.get_device_capability(0))

# a = torch.randn(1000, 1000, device="cuda")
# b = a @ a
# print("GPU operation successful:", b.shape)
# PY

# Step 6 — Pre-download the CIFAR-100 dataset
echo "Pre-downloading CIFAR-100 dataset..."
DATA_ROOT="$HERE/../data"
mkdir -p "$DATA_ROOT"

.venv311/bin/python -c "
from torchvision.datasets import CIFAR100
print('Downloading train split...')
CIFAR100('$DATA_ROOT', train=True, download=True)
print('Downloading test split...')
CIFAR100('$DATA_ROOT', train=False, download=True)
print('Dataset download complete.')
"

# Save environment snapshot
uv pip freeze --python .venv311/bin/python > environment-installed.txt

printf '\nEnvironment ready. On an allocated GPU node run:\n  MAX_SECONDS=300 ./run.sh\n'



mv .venv311/ ../
mv data/ ../

