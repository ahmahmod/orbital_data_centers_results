# E1_a_new: rootless directional network measurements

This version removes tcpdump and all privileged PCAP scripts. It **does not need sudo-g5k**. It preserves training (`train.py`), original step/collective CSVs, GPU/resource analysis, and plots.

## Run

```bash
cd ~/experiments_grid/synthesized_contact_plan/E1_a_new
N_NODES=4 GPUS_PER_NODE=1 MEASURE_TCP=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO MAX_SECONDS=240 ./run.sh
```

`NCCL_IB_DISABLE=1` explicitly selects the TCP-observable experiment mode rather than an RDMA baseline. NCCL may still use shared memory or GPU P2P within a physical host, neither of which is measured by `ss`. `NCCL_TRANSPORT_MODE=tcp_observable` disables P2P/SHM too, changing performance; label this run separately. If measuring native RDMA performance, do **not** interpret `ss` zero as no communication. All MPI/PyTorch/NCCL implementation/version dependencies stay unchanged.

`N_NODES` and `GPUS_PER_NODE` determine worker count dynamically. The default setup is 4 nodes × 1 GPU. Local GPU rank 0 launches `socket_counter_watch.py` from inside its already-running training process (one watcher per host), using the existing Python environment and `ss -tinp`. No extra `oarsh` launcher is created, and there is no shared monitor stop marker. Each sample records cumulative kernel TCP payload counters (`bytes_sent`, `bytes_received`, `bytes_acked`), source/destination addresses and ports, owner PID, worker rank, and node-wide NIC RX/TX counters. The script records timing and health even when no socket owner is visible.

## Result files

- `tcp_socket_samples_<host>.csv`: sampled TCP process-owned socket counters. A missing `bytes_sent` means the kernel did not expose the counter.
- `tcp_monitor_samples_<host>.csv`: sampling duration/accuracy, `ss` errors, map availability, and node-total NIC counters.
- `tcp_rank_pair_intervals.csv`: **observed TCP payload byte increments** between two consecutive samples, timestamp bounds, source/destination rank, ports, measured mean Mbps. Destination rank may be `unknown` where several GPUs share its IP and reverse socket ownership was not captured.
- `tcp_rank_pair_summary.csv`: directional sampled TCP volumes. Misses data before the first sample, after the last, or from very short-lived sockets.
- `tcp_rank_pair_windows.csv`: 100ms and 1s bins using **end-of-observation timestamps**. They are not packet-level traffic traces; if the sampler runs slower than the bin interval the apparent peaks can be inflated. Use `tcp_rank_pair_intervals.csv` for defensible interval rates.
- `tcp_nic_summary.csv`: node-global interface counters, including non-training traffic, useful only as an aggregate cross-check.
- `tcp_measurement_notes.json`, `socket_monitor_health.json`: assumptions, unknown attribution, sampling coverage and failure flags.
- `nccl_observability.json`, `observed_nccl_paths.csv`: NCCL runtime/version hints and logged channel topology **only**. No exact directional NCCL transport byte totals are claimed.
- Original `collectives_rank*.csv`, `steps_rank*.csv`, `traffic_demand.json`, and GPU metrics: logical DDP demand, not physical transfer volumes.

## NCCL-level directional transfer metrics

A fully quantitative NCCL `src_rank → dst_rank → completed transport bytes` measurement is **not provided by standard PyTorch DDP communication hooks or NCCL_DEBUG logs**. For true NCCL peer-level byte accounting, a compiled and runtime-version-compatible NCCL network/profiler plugin (or instrumented NCCL) would be necessary and must be tested across transport backends. This version intentionally reports that measurement as unavailable rather than guessing it from AllReduce bucket sizes. Socket-based TCP bytes are the operational alternative under this Grid'5000 reservation.

## Validation and caveats

1. Run `ss -tinp state established` while training runs; verify `bytes_sent` and `users:(...,pid=...)` are visible to your user. Some systems omit counters or process information, in which case `unknown` is expected.
2. Check `socket_monitor_health.json` for all hosts and no `ss` errors; check `tcp_measurement_notes.json` for stream/attribution coverage.
3. Capture is sampled, not packet-level. Socket counter resets are skipped; connection reuse can cause undercounts, and monitoring itself adds CPU load. Report `TCP_SAMPLE_PERIOD_S` (default 0.1 sec).
4. TCP bytes are payload bytes, not IP/Ethernet on-wire bytes; retransmissions and transport overhead are not fully represented. TCP counters also include training-process control/data messages (not NCCL-only).
5. If no positive per-pair deltas are observed, the run has **no usable TCP byte evidence**—not zero network demand.

## Installation

Back up your existing E1_a_new directory, then replace only the scripts in the supplied ZIP. Previous results are not included or modified. Do not mix old `traffic_*.pcap` or `capture.stop` results with the rootless output.

## Grid’5000 reliability changes (2026-11-06)

* **Fixed remote monitor startup:** the previous launcher started `oarsh` monitor processes before training and aborted when an immediate readiness check failed (sometimes with blank logs). Monitoring is now started by each node's local-rank-0 training worker after DDP initializes and `worker_map.json` becomes available.
* **No capture or monitor global stop markers:** `capture.stop` and `monitor.stop` are not used. Each host uses `monitor_stop_<short-hostname>`, written during its worker's `finally` block.
* **Best-effort TCP monitoring:** a monitor startup failure no longer prevents distributed learning. The analyzer and `socket_monitor_health.json` describe any missing or incomplete observations.
* **No root needed:** socket monitoring uses `ss`, while NIC counters read `/sys/class/net/<iface>/statistics/`.
* **No exact NCCL per-peer byte claims:** all TCP bytes are sampled counter differences and might include control traffic. Runtime logs show topology only.

After a run, check `node_*.log`, `monitor_launch_<host>.log`, `socket_monitor_health.json`, and `tcp_measurement_notes.json`.
