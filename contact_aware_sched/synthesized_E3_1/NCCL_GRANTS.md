# E3_1 low-overhead native NCCL contact grants (v4)

DDP uses native PyTorch DistributedDataParallel (broadcast_buffers=False),
with the same bucketed backward in OFF and AWARE. DiLoCo uses its explicit
delta all-reduce at fixed H=8 or H=64. All NCCL initialization finishes
before tc replay. NCCL is never destroyed or recreated between contacts.

## Batched admission

Rank 0 selects a feasible contact after all READY records, checks replay
application records, proposes a batch, collects ACKs and recomputes the
budget before publishing a common GRANT. All four ranks execute exactly
K operations in collective order, with no filesystem control rendezvous
inside the grant. Each operation still logs local metrics. DONE is sent
once after the last operation's CUDA and optimizer completion. At that
boundary, all ranks drain before another grant. Never cancel an active grant
on a local clock. The coordinator stops all workers at a common boundary.

K is the number of synchronization operations, NOT local steps H.
DDP gates before forward (potential reducer communication); no flattening
or explicit gradient all-reduce. DiLoCo prepares its first delta locally
before requesting a grant; subsequent operations in that grant include H
local steps each. Validation is deferred until a batch is drained, never
inside an active grant. Fixed epoch budget caps K, and a final partial
DiLoCo block is not globally aggregated.

The budget includes a trace-based transfer estimate, local compute floor,
a rolling 90th percentile of recent operation duration (worst rank, similar
rate/delay, 32 samples, 120 s expiry), safety factor, launch guard and margin. The first DDP
grant is K=1; initial DiLoCo grants are capped at two until a full-round
interval is observed. Old samples expire while waiting, so a single slow operation does not
permanently block admission. This remains an estimate, not a guarantee. Near closure K decreases; insufficient budget
means wait for the next window. No new grant is issued after the horizon.
OFF uses ReactiveScheduler, not BatchScheduler. It never evaluates contact
windows or performs READY/PREPARE/ACK/GRANT/DONE. The trace is loaded for
metadata and network replay is unchanged. Rank zero publishes one STOP-only
checkpoint every E3_REACTIVE_STOP_CHECK_OPS operations (default 32); peers
read that decision, without a mutual rendezvous. The wait timeout is at
least NCCL timeout + 60 s. No independent wall-clock exits are allowed.
The horizon can be exceeded by up to the remaining checkpoint operations
(and their network stalls); the runner retains its hard shutdown grace.
Unexpected NCCL failures remain fatal; no
mid-collective cancellation/restart is attempted. The runner terminates peers.

## Controls (forwarded to remote workers)

| Variable | Default | Purpose |
|---|---:|---|
| E3_GRANT_MAX_OPS | 32 | Maximum K; set to 1 for per-operation ablation |
| E3_LOCAL_STEP_BUDGET_S | 0.25 | Conservative local step floor |
| E3_OBSERVED_FACTOR | 1.5 | Safety multiplier for observed operation duration |
| E3_OBSERVED_WINDOW | 32 | Recent samples per rank |
| E3_OBSERVED_MAX_AGE_S | 120 | Expiry even while waiting |
| E3_OBSERVED_QUANTILE | 0.9 | Nearest-rank percentile |
| E3_NCCL_TOPOLOGY | ring_auto | Verified ring union or full_mesh |
| E3_CONTROL_HOST | frontend FQDN | Override reachable frontend address |
| E3_MIN_TRANSFER_S | 0.5 | Minimum trace-based transfer estimate |
| E3_MARGIN_S | 15 | Closing margin |
| E3_CONTACT_OPEN_GUARD_S | 0.25 | Opening guard |
| E3_LAUNCH_GUARD_S | 0.25 | Launch allowance |
| E3_CONTROL_POLL_S | 0.05 | Boundary polling |
| E3_RENDEZVOUS_TIMEOUT | 180 | AWARE missing rank timeout |
| E3_REACTIVE_STOP_CHECK_OPS | 32 | OFF sparse termination interval |
| E3_REACTIVE_STOP_TIMEOUT_S | 0 | Optional OFF stop-decision timeout floor |

A smaller transfer floor may be appropriate after measuring NCCL on actual
GPU nodes and emulated link rates. Aware runs may still fail if a grant
overruns a contact; tune from evidence, not by hiding failures.

## Testing

```sh
python3 -m unittest discover -s tests -p 'test_nccl*.py' -v
python3 -m unittest discover -s tests -p test_e3_ddp_gate.py -v
python3 -m py_compile e3_batch_control.py nccl_train.py runner.py
```

Inside a prepared four-GPU reservation, use a trace with sufficient feasible
contacts and actual outages in the selected horizon:
```sh
CASE=off MAX_SECONDS=120 DATASET=synthetic ./ddp/run.sh
CASE=aware E3_GRANT_MAX_OPS=1 MAX_SECONDS=120 DATASET=synthetic ./ddp/run.sh
CASE=aware E3_GRANT_MAX_OPS=32 MAX_SECONDS=120 DATASET=synthetic ./ddp/run.sh
CASE=aware E3_GRANT_MAX_OPS=32 MAX_SECONDS=120 DATASET=synthetic METHOD=diloco8 ./diloco/run.sh
```
Repeat for DiLoCo64 and continuous connectivity. Use identical trace,
training profile for the paired OFF/AWARE comparison; compare
K=1 vs K=32 separately as an ablation. Do not mix v1 explicit-DDP results
with v4 native-DDP results or v2/v3 results. The old
per-operation coordinator remains for legacy tests; AWARE uses BatchScheduler
even for K=1, while OFF ignores K.

## Measurement

In AWARE each rank writes e3_batch_rankN.csv with K, H, grant and drain timestamps,
contact waiting, approximate coordination time excluding planned contact
waiting, observed duration, and budget overrun. Existing per-operation
scheduler and progress CSVs remain available in both modes. OFF creates no
batch CSV or grant folders; its sparse termination decisions are stored in
.e3_control/reactive_stop. The frontend TCP broker is the out-of-band control path; grant/stop files
are mirrored for diagnostics, not used for live worker polling. Native DDP backward duration
includes computation and NCCL overlap, not pure network time. Validation
time can vary across ranks outside grants and is not part of the active
batch estimate. The nominal wall budget is checked at shared boundaries;
a DiLoCo local block can extend beyond it without further NCCL admission
in AWARE; OFF can exceed the horizon until its next stop checkpoint.

## v4 control and topology

The uploaded AWARE run showed repeated 30/60-second control delays and a
798.5-second grant wait after step 11. The runner now hosts an authenticated
TCP broker on the frontend, preflight-tested from every worker before tc
shaping. AWARE grant control, live tc replay acknowledgments, and OFF sparse
stop decisions use it; no per-batch worker NFS visibility dependency. Startup
files and metric CSVs still use the shared filesystem. No silent production fallback.

Both OFF and AWARE explicitly use NCCL_ALGO=Ring and collect INIT/GRAPH/NET
logs before starting the trace. The scheduler uses the union of all verified
channel ring edges plus reverse TCP ACK edges. If channel logs are incomplete
or inconsistent, it safely uses all 12 directed edges. Multiple rings can also
produce all 12 edges. E3_NCCL_TOPOLOGY=full_mesh keeps Ring but uses the
conservative policy for an ablation. Compare fresh OFF/AWARE runs with the
same forced algorithm; old auto-algorithm results are not a matched baseline.
Logs do not prove that every possible auxiliary NCCL traffic flow uses only
the detected edges; validate on GPUs. The 15 s closing margin is retained;
a 5 s margin is a diagnostic option only. AWARE drain timeout is at least
NCCL timeout + 60 s, subject to the runner hard shutdown limit.

Rank 0 prints WAIT_READY, WAIT_CONTACT (with next contact and estimate),
WAIT_REPLAY, PREPARE, GRANT, DRAIN and DONE. The batch CSV records rate,
recent estimate and topology. For example:

```sh
E2_CONFIRM_SHARED_TC=1 CASE=aware E3_NCCL_TOPOLOGY=ring_auto E3_GRANT_MAX_OPS=32 E2_LOG_EVERY=1 MAX_SECONDS=900 ./ddp/run.sh
E2_CONFIRM_SHARED_TC=1 CASE=off E3_NCCL_TOPOLOGY=ring_auto MAX_SECONDS=900 ./ddp/run.sh
```

The new transport_outage_exposure_rate measures completed collective
interval overlap with verified ring/ACK edges and is null when topology
is unverified. The original all-pair onset metric is retained separately.
