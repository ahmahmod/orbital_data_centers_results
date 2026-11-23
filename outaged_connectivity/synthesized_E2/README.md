## Update: automatic isolated/filtered network selection

See [NETWORK_MODES.md](NETWORK_MODES.md). E2 now falls back to filtered peer-IP tc when no safe secondary KaVLAN NIC exists. On a shared NIC, applying outages requires `TC_APPLY=1 E2_CONFIRM_SHAPING=1 E2_CONFIRM_SHARED_TC=1`. This does not guarantee SSH safety.

# E2 — Four GPU satellite contact replay, self-contained source package

**Read this before using.** This package contains all of its own scripts, training workload, trace, logging and plotting code. It does **not** download code from sibling E1 directories, use a patcher, or require their scripts. It requires a Grid'5000 account, four available GPU nodes on one site, the usual GPU runtime dependencies, and CIFAR-100 (downloaded on first setup). No live Grid'5000 GPU job has been performed during package construction.

## Relation to original repository

- Source reviewed: `experiments_grid/emulated_dist_ai/cont_run.sh` (deployment, root SSH, KaVLAN, DHCP, route guard)
- Original E2 model families: DDP, ring gossip 8/64 steps, DiLoCo-style local SGD 8/64, FedAvg 8/64
- Original 4-worker 60-minute synthetic contact intervals were preserved by `generate_trace.py`.
- **Methodology difference:** `fl8/fl64` use Flower 1.24 legacy FedAvg server/client runtime (`train_flower.py`), but are freshly written rather than exact copies of E1 Flower code. DiLoCo/Gossip/DDP are also fresh implementations with a similar research algorithm; **do not treat their results as exact reproductions of E1**. The other workloads are fresh implementations with consistent ResNet18+CIFAR100 data and independent rank telemetry.

## GPU profiles (4 physically separate hosts, local VLAN only)

| Site | Default cluster | GPUs/host | Data interface | KaVLAN port | Queue |
| --- | --- | --- | --- | --- | --- |
| Lille | chifflot (override `CLUSTER=chuc` for A100) | 4x A100 40GB | ens15f1np1 | eth1 | default |
| Grenoble | kinovis | 2x L40S 45GB | ens15f1np1 | eth1 | default |
| Nancy | gruss | 2x A40 45GB | eno33 | eth2 | abaca |
| Nancy fallback | graffiti | 4x RTX 2080Ti | ens4f0 | eth2 | abaca |

The site profiles come from published Grid'5000 hardware inventory. `reservation.sh` requests 4 hosts from *one cluster* but cannot guarantee their real-time availability. Verify all actual NICs before `--apply`. Site names must not be mixed inside `kavlan-local`.

## Quick start — DEPLOY reservation (required for KaVLAN assignment)

This version follows `emulated_dist_ai/cont_run.sh`: reserve a **deploy** job on a Grid'5000 frontend, perform `kadeploy3 -k`, access nodes with `ssh root@host`, assign isolated Ethernet ports to KaVLAN, run DHCP and check all 12 worker routes.

1. On the **Lille frontend**: `SITE=lille ./reservation.sh` reserves four Chifflot GPU hosts + `kavlan-local` using `-t deploy` and the `abaca` queue. Override `CLUSTER=chuc` when four Chuc hosts are available. For another site, set `SITE=grenoble` or `SITE=nancy` on its frontend.
2. In the resulting frontend OAR job shell: `E2_CONFIRM_NETWORK=1 ./setup.sh` deploys the selected image and configures KaVLAN. Default image: `debian12-x64-nfs`. To use a GPU-enabled image: `ENVIRONMENT=<installed-GPU-image> E2_CONFIRM_NETWORK=1 ./setup.sh`.
3. Setup checks NVIDIA, node SSH, the isolated NIC, DHCP, and 12 directed routes. A default Debian environment may **not** have NVIDIA drivers: in that case setup will stop at GPU runtime verification after configuring the network. Select/configure an appropriate image. `E2_REDEPLOY=1` explicitly redeploys and wipes the prior node OS.
4. When all four nodes have NVIDIA and see the NFS source directory, install dependencies into shared home if required: `./dependencies.sh` (install must use a compatible Python on deployed nodes). Test separately before a full run.
5. `DATASET=synthetic METHOD=gossip8 MAX_SECONDS=180 ./run.sh` tests without `tc` shaping. To enable schedule impairment use `E2_CONFIRM_SHAPING=1 TC_APPLY=1 METHOD=gossip8 MAX_SECONDS=180 DATASET=synthetic ./run.sh`.

`./run.sh` does **not** redeploy the nodes; call `setup.sh` once per new reservation. Re-running a destructive OS deployment during training would invalidate the experiment.

## Other notes

- KaVLAN commands execute on the frontend (or through SSH to the frontend) with `-j $OAR_JOB_ID`.
- Do not change `br0` or the management NIC. The default VLAN is local to one site and cannot span Lille, Nancy and Grenoble.
- The `sites.json` Linux NIC/KaVLAN port profiles must be verified against the actual hardware inventory before applying.
- Keep the original job shell open, and ensure enough walltime for OS deployment, dependency installation and the 60-minute experiment.

## What is / is not emulated

Each host has one worker, one NVIDIA GPU, and three per-peer egress flows. On all hosts `tc` conditions direct-link bandwidth, propagation delay, and 100% packet loss during contact outages. This simulates scheduled per-peer network impairment, **not optical hardware or a physical orbital constellation**. Routing is not automatically redirected through intermediate workers if a direct link is down. The trace uses symmetric per-pair schedules and contains 12 directed flows.

The trained ResNet18 uses CIFAR-100, SGD, batch size 32; the same local model setup is used for all methods. Contact plan is not given to the optimizer. This is an E2 behavior comparison, not a proactive E3 scheduler. Stalled DDP or NCCL may fail during outages; a failure is a valid result and is recorded, not suppressed.

## File layout

- `reservation.sh`: reserve one local VLAN and 4 physical GPU hosts from a known cluster; use `SITE`/`CLUSTER` to choose.
- `sites.json`: explicit verified hardware profile and supported data NIC.
- `remote.py`: standard-mode `oarsh` or deployed root `ssh`; route separation checks.
- `bootstrap.py`, `bootstrap.sh`: based on `emulated_dist_ai/cont_run.sh` with four-node NIC/KAVLAN/DHCP/route checks.
- `inspect.sh`: read-only hardware and route inspection.
- `dependencies.sh`: install PyTorch CUDA, torchvision, Flower 1.24 and plotting dependencies into E2's own `.venv`.
- `download_cifar.py`: obtain third-party CIFAR-100 dataset locally.
- `train.py`: DDP, gossip and DiLoCo variants; `train_flower.py`: Flower 1.24 FedAvg server and clients.
- `tc_agent.py`: replay, per-destination egress shaping with cleanup.
- `runner.py`, `run.sh`: coordinate four training processes and four `tc` agents.
- `analyze.py`: event correlation, outage-step progress and figures.
- `generate_trace.py`, `traces/directed_trace_4workers.csv`: deterministic synthetic schedule.
- `cleanup.sh`: inspect or explicitly remove E2 HTB shaping.

## Risks and constraints

- KaVLAN membership/remote privilege semantics depend on site and job type; inspect actual resources before enabling them.
- `sudo-g5k` permissions for `dhclient` and `tc` are not guaranteed; under standard jobs the script will fail clearly if unsupported.
- NVIDIA driver/torch CUDA ABI must match actual nodes. The installation script cannot guarantee network/package-index access at all sites.
- Four hosts must see a consistent **NFS-mounted package path, Python environment, and dataset**. The runner explicitly tests this, rather than assuming it silently.
- NCCL rank initialization uses the isolated IPv4 address and `NCCL_SOCKET_IFNAME`. Intra-node GPU traffic is not part of the contact plan.
- E2's results are comparable within this standalone package, but should not be claimed as reproduction of the existing E1_b/Flower sources.
- Each run lasts up to `MAX_SECONDS` of training plus `STARTUP_LEAD` and cleanup; arrange enough OAR walltime.

If a node fails GPU, NIC, shared NFS or route checks, do NOT enable shaping. Fix the reservation/setup instead. `E2_CONFIRM_CLEANUP=1 ./cleanup.sh` removes E2's `1:` HTB root qdisc on the experimental NIC only.


## Five-hour repeated contact plan and three-hour E2 runs

The checked-in `traces/directed_trace_4workers.csv` covers **18,000 seconds (5 hours)**.
It repeats the original deterministic 3,600-second synthetic contact schedule five
consecutive times, retaining all 12 directed edges, bit rates, propagation delays,
and per-pair outages. This is **not** an orbit-derived contact plan.
Rebuild with `python3 generate_trace.py`.

`./auto.sh` now runs **10,800 seconds (3 hours)** per method by default,
with CIFAR-100 and a 300-second validation interval. The extra two hours of trace
are available for longer trials through `DURATION=18000 ./auto.sh`, provided the
reservation and runner have enough walltime. The batch skips a previous attempt
only if its method, duration, dataset, trace hash and tc-enabled configuration
all match; previous one-hour results will not be mistaken for three-hour trials.
Run `python3 -m unittest discover -s tests` before deployment. Seven three-hour
runs require **at least 21 hours plus startup and cleanup**, spread across OAR
reservations if necessary. `auto.sh` checks available time before starting each
method. A failed run with a matching config is still considered attempted; move
its result folder aside before explicitly retrying.
