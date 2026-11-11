# E1_b — DiLoCo-style Local SGD and Asynchronous Gossip (pure PyTorch)

**Reference:** `synthesized_contact_plan/E1_a_new` (not `E1_a`). Its completed source code, `plot.py`, GPU sampler, rootless `ss` TCP monitor, and TCP analysis scripts remain unchanged. E1_b copies them locally using `install.sh`, without overwriting or modifying `E1_a_new`.

This is an **initial research implementation**, tested with two CPU/Gloo processes and synthetic data; actual Grid'5000 4-GPU/CIFAR-100 execution is **not yet verified**. The DiLoCo implementation is explicitly **DiLoCo-style local SGD with averaged pseudo-gradients and separate outer momentum**, not a claim of byte-for-byte reproduction of the original DiLoCo paper. Gossip is a directed asynchronous *ring* using background Gloo send/receive threads and full-model snapshots, not push-sum or a proven optimization algorithm. Both use stable links (no satellite trace): E1_b measures **communication demand** before E2.

## Install in your Git repository

```bash
cd ~/experiments_grid/synthesized_contact_plan
# Extract the downloaded E1_b.zip here to create E1_b/
cd E1_b
chmod +x install.sh run.sh
./install.sh
```

`install.sh` copies the exact checked-out `E1_a_new/train.py` as `e1a_reference.py` and the following **same measurement tools** into E1_b: `plot.py`, `socket_counter_watch.py`, `analyze_socket_counters.py`, `socket_monitor_health.py`, `plot_tcp_counters.py`, `nccl_diagnostics.py`, `selftest_rootless.py`. It never changes the E1_a_new source or results. Commit the installed files if you want a fully self-contained reproducible snapshot.

## Run after existing OAR reservation (4 GPU ranks)

Defaults: `N_NODES=4`, `GPUS_PER_NODE=1`, `NET_IFACE=br0`, `../.venv311/bin/python`, `../data`, CIFAR-100 45k/5k stratified holdout, ResNet18 CIFAR stem, SGD warmup/cosine, seed 2026, batch 32 per rank, 3600-second epoch-boundary budget.

```bash
METHOD=diloco LOCAL_STEPS=32 OUTER_LR=1.0 OUTER_MOMENTUM=0.9 MAX_SECONDS=300 ./run.sh
METHOD=gossip GOSSIP_EVERY=32 GOSSIP_ALPHA=0.5 MAX_SECONDS=300 ./run.sh
```

Full hour:

```bash
METHOD=diloco LOCAL_STEPS=32 MAX_SECONDS=3600 ./run.sh
METHOD=gossip GOSSIP_EVERY=32 MAX_SECONDS=3600 ./run.sh
```

Two nodes x two GPUs per node: `N_NODES=2 GPUS_PER_NODE=2 METHOD=gossip ./run.sh`.

`MAX_SECONDS` is checked at epoch boundaries, like E1_a_new. A 1-hour measurement needs additional reservation time for data initialization, validation, and plotting. No new venv, dataset download, KaVLAN, or `tc` required.

## Training algorithms

- **DiLoCo-style:** each rank optimizes its local complete ResNet18 for H `LOCAL_STEPS`; synchronizes pseudo-gradient `theta_anchor - theta_local` through `torch.distributed.all_reduce`; updates a separate outer-momentum buffer and replaces the local model with the common outer parameters. We flush a partial H-block at every epoch boundary for common validation. This produces `outer_allreduce` events.
- **Async gossip:** each rank optimizes locally every step. Every `GOSSIP_EVERY` steps it enqueues a full float32 model snapshot for background directed Gloo/TCP transmission to its right-hand neighbor `(rank+1)%world`; concurrently a background thread receives from `(rank-1)%world`. The training thread polls arrivals at safe optimizer-step boundaries and mixes `theta <- (1-alpha)*theta + alpha*theta_incoming`. It neither waits for each peer message nor performs a per-message all-worker barrier. Queue back-pressure may skip sends. There remain **epoch-boundary** barriers, validation, and stop broadcast to match E1_a_new's reporting. Training is therefore asynchronous **within epochs**, not a fully barrier-free system.

## Same metrics as E1_a_new

Preserved *exact CSV column names* from the checked-out E1_a_new for:

| Existing output | Captures |
|---|---|
| `steps_rank*.csv` | loss, LR, epoch, step, local/global sample counts, data wait/H2D/forward/backward/optimizer/other/total timing, samples/sec, GPU allocation, collective-event summaries |
| `validation.csv` | validation loss/accuracy, wall-clock, samples, time-to-target source |
| `sync_events_rank*.csv` | initialization, epoch barriers, stop broadcast, final barrier |
| `resources_rank*.csv` | GPU utilization, memory, power, NIC RX/TX counters |
| `collectives_rank*.csv` | logical communication operation submissions/completions and sizes (method-labeled) |
| `tcp_socket_samples_*.csv` | `ss` TCP counters, rank/endpoint attribution where observable |
| `tcp_monitor_samples_*.csv` | monitor health, coverage and NIC snapshots |
| `tcp_rank_pair_*`, `tcp_nic_summary.csv`, `socket_monitor_health.json`, `tcp_measurement_notes.json` | E1_a_new analysis with original limitations |
| `summary.json`, `manifest.json`, `worker_map.json`, `checkpoint_latest.pt`, `plots/*.png`, `analysis.json` | same top-level reporting and figures |

In addition, `messages_rank*.csv` reports exact application *logical* directed rank-to-rank gossip sends, receptions, missed/skipped sends, staleness versions, and DiLoCo periodic outer exchanges. `demand_b.py` generates `traffic_demand.json`, `p2p_pair_summary.csv`, and additional communication plots.

**Scientific caveat:** `backward_with_comm_s` is pure local backward in E1_b (E1_a_new DDP includes overlapped AllReduce). For E1_b, `collectives` columns count completed **method-specific events** in that step, including gossip sends which are *not collectives*. They are retained for schema compatibility and disambiguated by `collective_membership`. A 44.9 MB `send_posted` is *not* proof that TCP delivered that amount: count only `send_complete` and cross-check TCP socket samples. `ss` sampling can miss messages or ownership, and TCP payload bytes include control traffic and may reflect a different scope from application payloads. NIC counters are node-wide, not per-process.

**Convergence caveat:** gossip's rank0 `validation.csv` measures only the rank0 local model; other workers' models differ and an averaged-model validation would be a separate metric. Neither local optimizer state averaging nor error feedback is done. Comparisons to E1_a_new should include accuracy-vs-samples and wall time and disclose these algorithmic differences. Both methods use the same data loader and split from E1_a_new.

## Check results

```bash
cat results/E1_b_*/summary.json
cat results/E1_b_*/traffic_demand.json
head results/E1_b_*/messages_rank0.csv
```

If the rootless TCP monitor lacks visibility, results remain valid for learning but are **not** evidence of zero network traffic. Check monitor-health and measurement-notes reports before interpreting physical bytes.

## Git commit (if you downloaded the package)

```bash
cd ~/experiments_grid
# Review changes before committing
 git status --short
 git add synthesized_contact_plan/E1_b
 git commit -m 'Add E1_b DiLoCo-style and asynchronous gossip benchmarks based on E1_a_new'
 git push origin HEAD
```

## Environment assumptions

The E1_a_new PyTorch environment and CIFAR-100 dataset must already be available to every reserved node via shared storage. For multi-host gossip, port/interface restrictions must allow `GLOO_SOCKET_IFNAME=br0` traffic. `NCCL_IB_DISABLE=1` retains the TCP-observable network setting from E1_a_new, but gossip model messages use **Gloo/TCP** rather than NCCL; do not claim the methods use identical communication runtimes.
