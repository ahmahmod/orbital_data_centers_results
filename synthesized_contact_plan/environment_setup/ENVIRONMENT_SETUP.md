# Grid’5000 E0: reproducible Python environment

**Runtime specification:** Debian 13 (Trixie), Linux x86_64; **Python 3.11** in `.venv311/`; **PyTorch 2.5.1 + CUDA 11.8**, **torchvision 0.20.1 + CUDA 11.8**. Compatible with Tesla P100/V100 in principle, conditional on a suitable installed NVIDIA driver. Only the GPU node provides actual CUDA access; the frontend can prepare files and download dependencies if it has internet access.

## Files
- `requirements.txt`: pinned direct Python dependencies and CUDA wheel index.
- `setup_env.sh`: creates `.venv311`, installs packages, tests imports and CUDA, records installed dependency versions.
- `run.sh`: existing E0 runner that defaults to `.venv311/bin/python`.

## One-time setup
Place `requirements.txt` and `setup_env.sh` **beside `run.sh`, `train.py`, `plot.py`** on a shared or persistent Grid’5000 filesystem.

```bash
# From the E0 directory (e.g. ~/tcp_exp/synthesized_contact_plan/E0)
export PATH="$HOME/.local/bin:$PATH"
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
chmod +x setup_env.sh run.sh
./setup_env.sh
```

If the compute node cannot download packages, execute setup on a frontend with a shared accessible filesystem. Ensure the environment path and interpreter work on the compute node: virtual environments should not be moved after creation.

## Verify on the reserved GPU node

```bash
nvidia-smi
./.venv311/bin/python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); assert torch.cuda.is_available()'
```

## Execute E0

```bash
MAX_SECONDS=300 ./run.sh
# Full run
MAX_SECONDS=3600 ./run.sh
```

## Reproducibility

`environment-installed.txt` records resolved transitive package versions. Include this file, `requirements.txt`, the GPU model, driver version (`nvidia-smi`) and the training manifest with published results. The pinned direct requirements are *not* a full transitive lockfile; the installed snapshot captures what was used for a specific run. Dataset must be available or downloadable (`train.py` uses CIFAR-100).

If installation reports that CUDA wheels are unavailable, verify Python is 3.11, the system is x86_64, and that outbound access to the PyTorch wheel index is allowed. `run.sh` should be executed only after the GPU verification succeeds.

Official references: https://docs.pytorch.org/get-started/previous-versions/ and https://docs.astral.sh/uv/guides/install-python/.
