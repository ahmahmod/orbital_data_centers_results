# E1_b — Flower FedAvg add-on (four clients / four GPUs)

This is a **separate FedAvg baseline** for `synthesized_contact_plan/E1_b`, not a replacement for DiLoCo or gossip. Add `train_flower.py`, `run_flower.sh`, `plot_flower.py`, `requirements_flower.txt` to the existing E1_b directory. Use **E1_a_new**, not E1_a, as the source of `e1a_reference.py` and the existing instrumentation/plotting utilities.

## Compatibility / dependencies

Uses the documented **legacy Flower gRPC `start_server`/`start_client` and `NumPyClient.to_client()` API**, pinned to **`flwr==1.24.0`** for a one-process-per-physical-GPU Grid'5000 deployment. That API is deprecated in modern Flower; the supported modern architecture uses ServerApp / ClientApp / SuperLink / SuperNode. This consciously pins a legacy API to avoid making the experiment depend on SuperNode deployment and a new application packaging model. Do not upgrade Flower without retesting.

On the same shared venv already containing torch, torchvision, numpy, matplotlib and CIFAR-100:

```bash
PYTHON=../.venv311/bin/python
"$PYTHON" -m pip freeze > e1b_environment_before_flower.txt
"$PYTHON" -m pip install 'flwr==1.24.0'
"$PYTHON" -c 'import flwr,torch,torchvision; print(flwr.__version__,torch.__version__,torchvision.__version__)'
```

This installs Flower (a new dependency) but **does not redownload CIFAR-100, reinstall CUDA, or recreate the venv**. Installing Flower may alter transitively required packages. Verify the previously working E1_b imports afterward.

## Run from an active OAR GPU reservation

Four hosts / one GPU per host:

```bash
chmod +x run_flower.sh
ROUNDS=10 LOCAL_STEPS=32 ./run_flower.sh   # short functional pilot
ROUNDS=100 LOCAL_STEPS=32 ./run_flower.sh  # full rounds-based experiment
```

Two hosts / two GPUs each:

```bash
N_NODES=2 GPUS_PER_NODE=2 ROUNDS=100 LOCAL_STEPS=32 ./run_flower.sh
```

If paths differ: `PYTHON=/path/to/shared/venv/bin/python DATA_ROOT=/path/to/data NET_IFACE=br0 ./run_flower.sh`.

`run_flower.sh` uses the existing reserved hosts via `oarsh`; it runs a CPU-based Flower coordinator on the first GPU node and launches **four Flower clients** (one GPU each). All hosts need the same shared source/output paths, CUDA environment, and an IP on `NET_IFACE`. The coordinator uses port 8085 by default (`PORT=...` to override). This uses **insecure gRPC**: use it only on an isolated trusted experiment network.

**IMPORTANT TIME NOTE:** This implementation terminates at `ROUNDS`, not at the exact value of `MAX_SECONDS`. The max-seconds field is recorded for cross-experiment documentation only. For a fixed 60-minute comparison, choose rounds after a pilot so the measured wall time is comparable, or implement a coordinated round-boundary time-stop extension; never report a fixed 1h budget that was not enforced.

## Model and data

- ResNet18 CIFAR stem, CIFAR-100 stratified training/validation holdout and transforms via `e1a_reference.data_loaders()`.
- Each Flower client uses the same *DistributedSampler partition* as E1_a_new, with the same seed and batch size.
- Every client performs `LOCAL_STEPS` local SGD steps before uploading its complete model to the Flower coordinator.
- The coordinator aggregates client model parameters using **FedAvg weighted by number of local samples**, then broadcasts the aggregated model for the next round.
- The local optimizer state resets at each communication round (explicit FedAvg choice). Learning rate follows the E1_a_new warmup/cosine schedule, keyed to each client's data-loader epoch.
- All clients evaluate the **same global model**, on the identical full validation holdout after every FedAvg round. This differs from the divergent local models of gossip.
- Because E1_a_new and FedAvg make different optimization/communication choices, accuracy-vs-time comparisons must also present accuracy-vs-processed-samples and clearly state the local-steps and global-batch differences.

## Metrics and results

### Same inherited E1_a_new / E1_b-compatible outputs

`steps_rank0.csv` ... `steps_rank3.csv`, `validation_rank*.csv`, `validation_aggregate.csv`, legacy `validation.csv`, `collectives_rank*.csv`, `sync_events_rank*.csv`, `messages_rank*.csv`, `resources_rank*.csv`, rootless `tcp_*` socket-monitor outputs, `manifest.json`, `summary.json`, rank-local checkpoints and `checkpoint_latest.pt`, original `plot.py` figures.

The existing CSV columns are retained. **Interpretation matters:** `backward_with_comm_s` for Flower is local backward, not DDP AllReduce; `collectives` counts *logical model exchange records* at the application callback boundary, not NCCL collectives. The precise gRPC transfer time is not measured by the client's callback. The inherited TCP socket monitor (when available) measures actual socket activity subject to its sampling/attribution limits. `collective_bytes_logical` and per-message bytes never represent physical wire bytes.

### Flower-specific outputs

`rounds.csv` (round-level model payload accounting and durations), `fit_clients.csv` (per-client local-training metrics), `40_flower_global_accuracy.png`, `41_flower_round_communication.png`, `42_flower_client_train_time.png`, `43_flower_aggregator_time.png`.

The coordinator's `checkpoint_latest.pt` is the **aggregated global model**. `checkpoint_rank*.pt` is the latest **pre-aggregation local client model**. This is unlike gossip where no global FedAvg model exists.

## Network / measurement caution

- Topology is **star**: GPU client ⇄ central Flower server, not peer-to-peer. Each round includes model download, upload, and a second global-model download for validation. The server typically sends model weights again at the next training round.
- `rounds.csv` logical byte estimates do **not** include gRPC headers, control traffic, retransmissions, compression/serialization overhead, or non-training communications.
- The rootless socket monitor is reused on GPU client hosts; on host 0 it may also observe coordinator traffic. NIC counters are node-level; attribute directional bytes using the actual socket records and server IP rather than assuming physical bytes equal logged tensor payload.
- Do not use the previous `demand_b.py` to interpret Flower `model_upload` as gossip P2P; use `rounds.csv` and rootless TCP counters.
- Flower 1.24 is a pinned experimental dependency. `flwr` was not available in the artifact-building runtime, so **real Flower/GPU execution has not been tested**. Python/Bash syntax and plotting are tested. Run a short four-GPU pilot first.
