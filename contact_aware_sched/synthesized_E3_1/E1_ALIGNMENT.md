# E2 E1-aligned training profile (November 2026)

This revision aligns the **shared training configuration** across E1_a DDP, E1_b DiLoCo/Gossip, E1_b Flower, and E2 while retaining E2's scheduled tc outage replay and time-based measurements.

| Setting | E2 aligned value |
|---|---|
| ResNet18 stem | 3x3, stride 1, no maxpool |
| Dataset | CIFAR-100 |
| Training split | E1_a stratified 45,000-image training / 5,000-image holdout from the 50,000 training images |
| Validation | Same fixed 5k holdout on all workers, local-model evaluation |
| Training augmentation | RandomCrop(32,padding=4), RandomHorizontalFlip |
| Normalization | mean=(.5071,.4867,.4408), std=(.2675,.2565,.2761) |
| Training partition | DistributedSampler with seed=2026, shuffle=True, 4 ranks |
| Per-rank batch | 32 |
| Number of data workers | 4 |
| Seed | 2026 |
| Optimizer | SGD, momentum=.9, weight_decay=.0005 |
| LR | .1 with five-epoch warmup, cosine decay to .0001 over 100 epochs |
| Label smoothing | 0 |
| Precision | FP16 autocast and GradScaler (disable using --no-amp for diagnostics) |
| Training window | 3600s, unchanged |
| Evaluation cadence | E2 remains wall-clock, default every 300s, without extra validation barriers |
| DDP | Native torch.nn.parallel.DistributedDataParallel with bucketed gradient reductions (as E1_a) |
| DiLoCo | one flattened NCCL all-reduce every 8/64 steps; outer_lr=1.0, momentum=.9 |
| Gossip | separate Gloo process group with CPU tensor ring transfers every 8/64 steps |
| Flower | four-client FedAvg, 8/64 steps per round, with SGD momentum reset each round |

## Differences that must still be documented

- E2 is time-limited and intentionally uses independent wall-clock validation; E1 samples by epoch and uses epoch barriers. These are not exactly the same synchronization semantics.
- E2 DiLoCo H=8/64 and Gossip H=8/64 intentionally differ from the E1_b launcher defaults H=32.
- E2 Gossip implements its own asynchronous ring scheduling and may not be identical to E1_b queue/timeout details.
- E2 and E1 use different networking and launcher orchestration. E2 uses contact-aware shaping; E1 does not.
- E2 Flower still uses its own client/server lifecycle and E2's wall-clock coordinator.
- DDP communication timing in E2 currently includes the entire backward pass, not only NCCL time; do not compare its recorded communication duration with E1_a's instrumented bucket timings without re-instrumentation.
- Old E2 result directories are not modified. Their metrics are **not directly comparable** with new E1-aligned runs.

## Verification before long runs

```bash
cd synthesized_contact_plan/filter_E2
python3 -m py_compile e1_training.py e2_metrics.py train.py train_flower.py runner.py
# Run inside an active, fully configured four-GPU OAR job:
METHOD=ddp DATASET=cifar100 MAX_SECONDS=180 TC_APPLY=0 E2_EVAL_INTERVAL_S=60 ./run.sh
```

The smoke run should verify initialization, data loading, AMP, the native DDP backward, validation and clean process shutdown. Validate the no-shaping run before enabling tc outages on the shared management NIC. E2 training changes have not been GPU-tested as part of this GitHub edit.
