# E0 — single-GPU, no-trace training reference

This is **E0**, NOT a distributed DDP run. It establishes a training/compute reference for E1 (two-worker DDP) and E2 (DDP + traces). This package was smoke-tested locally with synthetic FakeData on CPU; no Grid'5000 GPU or real CIFAR-100 experiment has been run by the author of this package.

## 1. Reserve a GPU on Grid'5000

On a site frontend, use a site/queue-appropriate reservation such as:

```bash
oarsub -I -l "gpu=1,walltime=2:00:00"
```

Grid'5000 documentation: https://grid5000.fr/w/GPUs_on_Grid5000 . Check site/queue access. `oarsub -I` places you on the reserved node, where you should execute the training. This does **not** need KaVLAN or `tc`.

## 2. Python environment

On the allocated GPU node, inspect `nvidia-smi`. Use an existing compatible PyTorch+torchvision environment from the cluster, or install compatible CUDA-enabled PyTorch and torchvision following https://pytorch.org/get-started/locally/ . Do not blindly install the CPU-only wheels. Matplotlib and NumPy are required for plotting.

```bash
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

The training script runs on CPU for smoke testing; for the primary E0 run **ensure the selected device is `cuda:0`** in `manifest.json`.

## 3. Prepare CIFAR-100

If the reserved compute node has Internet access, initialize the dataset once:

```bash
python3 -c 'from torchvision.datasets import CIFAR100; CIFAR100("./data",train=True,download=True); CIFAR100("./data",train=False,download=True)'
```

Otherwise, download it on a machine with access and copy the `data/` directory to your experiment directory on the reserved node. CIFAR-100 is ~170 MB compressed. Subsequent training runs use the cache without `download=True`.

**Note:** Here CIFAR-100's official test split is used as the *demonstration validation set*, because this is a quick baseline. For publishable time-to-accuracy claims, use a train/validation split from the original training set, reserve the official test set for final evaluation, and freeze the split across E0–E8.

## 4. Smoke test (not a research measurement)

```bash
python3 train.py --output results/smoke --dataset fake --cpu --num-workers 0 \
  --train-subset 16 --val-subset 8 --batch-size 4 --max-steps 2 --eval-every-steps 2
python3 plot.py results/smoke
```

## 5. Main E0: single GPU, CIFAR-100, ResNet-18

Run an initially short 5-minute pilot:

```bash
MAX_SECONDS=300 EPOCHS=20 ./run_e0.sh
```

Then the 1-hour evaluation:

```bash
MAX_SECONDS=3600 EPOCHS=100 RUN_ID=E0_seed2026_1h ./run_e0.sh
```

By default: 32 batch size, SGD learning rate 0.1, momentum 0.9, weight decay 0.0005, seed 2026, validation every 200 steps, one process / GPU. Stop at the time budget or completed epochs. Budget is measured from the beginning of the training loop, excluding dataset/model setup; checkpointing and validation add modest time. A 2-hour OAR allocation is safer for downloading, setup and a 1-hour run.

For a run with a specified validation-accuracy stop (fraction, e.g. 0.5):

```bash
MAX_SECONDS=3600 EPOCHS=100 RUN_ID=E0_accuracy ./run_e0.sh --target-accuracy 0.50
```

Target accuracy must be declared before comparative E1+ trials. This particular run's accuracy target is illustrative only.

## 6. Output files

```
results/<run_id>/
  manifest.json              host, software, GPU, seed, parameters and caveats
  steps.csv                  time/loss/step timing/throughput/memory measurements
  validation.csv             accuracy/loss, elapsed time and samples seen
  gpu.csv                    NVIDIA GPU utilization/memory/power samples (if available)
  checkpoint_latest.pt       model + optimizer state at last evaluation
  summary.json               stop reason, steps, elapsed time, final validation
  console.log                complete stdout/stderr of training
  plots/
    01_train_loss.png
    02_validation_accuracy.png
    03_validation_loss.png
    04_step_duration.png
    05_samples_processed.png
    06_gpu_utilization.png   if nvidia-smi produced samples
```

Run `python3 plot.py results/<run_id>` to regenerate plots. All plots are derived from observed CSV records; none is projected/estimated.

## 7. Measurements and comparability

- Elapsed clock starts just before the first training batch fetch; CIFAR-100 preprocessing and model initialization are outside that budget.
- All `step_s`, `forward_s`, `backward_s`, `optimizer_s` are wall-clock intervals with CUDA synchronization for accuracy. This adds timing overhead and must stay consistent in E1–E8.
- `data_s` includes waiting for the training batch. `samples_per_s` is step-local, while validation rows' running throughput includes validation overhead.
- `sync_s=0` by definition in E0: there is no distributed gradient communication. **This is not a measured synchronization latency.**
- GPU utilization is sampled via `nvidia-smi` (device-wide coarse sample), not directly equivalent to GPU idle time or a per-process timeline.
- `gpu_memory_allocated_mb` is PyTorch tensor memory, not `nvidia-smi` allocated device memory.
- Report training accuracy/loss and validation accuracy/loss separately; compare validation accuracy **against wall clock and samples processed**.
- For statistical comparisons use same workload, random seeds, augmentation policy, data partitions, machine class, training budget and time-to-accuracy target across E1–E8; run multiple seeds.
- E0 is a useful single-worker reference but is not a network-adaptation baseline. E1 and E2 will provide the actual DDP network baselines.

## 8. Future E1+ extension plan

Reuse exact ResNet-18 CIFAR-100 preprocessing, optimizer setup, CSV schema and validation definitions. In E1, add process-group initialization, DistributedSampler, per-rank log files, a clearly measured gradient synchronization interval, world_size=2 and `torchrun` on the two Grid'5000 nodes. E2 reuses E1 unchanged while `tc/netem` replays the schedule. For C1/C2, introduce controlled policies and log each scheduling decision. Do **not** claim E0 alone establishes a benefit for schedule awareness.
