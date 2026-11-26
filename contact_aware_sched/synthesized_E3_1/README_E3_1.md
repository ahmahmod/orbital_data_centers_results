# E3_1 — Contact-aware synchronization scheduling (contribution 1)

This folder is a **snapshot/copy of** `outaged_connectivity/synthesized_E2` on
`main` at commit `717bcee987b78486b82c9b50f659e38ff2e14b28`, with
new contact-aware scheduling and extended C1/C2 measurement files. E2 is untouched.

## Layout

```text
contact_aware_sched/synthesized_E3_1/
├── ddp/run.sh           # DDP: off, aware or both
├── diloco/run.sh        # METHOD=diloco8|diloco64
├── gossip/run.sh        # METHOD=gossip8|gossip64
├── flower/run.sh        # METHOD=fl8|fl64
├── run_tech.sh          # common paired-run controller
├── run.sh               # inherited E2 safety and Grid'5000 launch logic
├── runner.py            # E2 replay/launch, new E3 args
├── train.py             # DDP + DiLoCo + gossip model code
├── train_flower.py      # Flower FedAvg
├── e3_schedule.py       # contact oracle, rendezvous, delay logging
├── e3_analyze.py        # C1 + C2 common metrics
├── compare_pair.py      # paired figures after both runs
├── tests/test_e3_schedule.py
├── reservation.sh       # exact Nancy graffiti-5,7,8,9 predicate
├── reservation_e2.sh    # untouched original E2 reservation
├── reuse_e2_runtime.sh  # optional reuse of E2 venv, data, job state
├── setup.sh             # inherited full network setup (requires permission)
├── tc_agent.py          # inherited replay and qdisc cleanup
└── traces/              # existing exact synthetic trace
```

## Technique-local results and comparisons

Each technique stores its own results, including both OFF and aware runs,
and comparison figures:

```text
ddp/{results/,comparisons/}
diloco/{results/,comparisons/}   # diloco8 and diloco64 distinguished in run names
gossip/{results/,comparisons/}   # gossip8 and gossip64 distinguished in run names
flower/{results/,comparisons/}   # fl8 and fl64 distinguished in run names
```

The technique `run.sh` scripts call `run_tech.sh`, which supplies explicit
`--out` paths under the correct technique. Direct `runner.py` calls without
`--out` also default to the same technique's `results/` directory.
`compare_pair.py` defaults to that technique's `comparisons/` directory,
while an explicit `--out` takes precedence. These generated directories are
ignored by Git. Existing run folders are not moved automatically.

## Technique-isolated training and communication policies

The E3_1 runner dispatches each method to its own **trainer and scheduler**:

| Method | Trainer | Policy |
|---|---|---|
| DDP | `ddp/train.py` | `ddp/schedule.py` — global gradient collective |
| DiLoCo H=8/64 | `diloco/train.py` | `diloco/schedule.py` — blocking outer synchronization |
| Gossip H=8/64 | `gossip/train.py` | `gossip/schedule.py` — peer-to-peer ring sends only |
| Flower FL-8/64 | `flower/train.py` | `flower/schedule.py` — all-client round upload gate |

`e3_schedule.py` is a shared, policy-neutral contact-trace reader and event
logger, not a training-method scheduler. Dataset/model helpers and the original
baseline reference files remain at the E3_1 root.

**Gossip C1 protocol correction:** Both OFF and aware runs now evaluate local
model accuracy at each epoch boundary **without pre/post global barriers or a
collective stop broadcast**. Their only scheduling difference is peer-send
admission. This deliberately changes the old E2 epoch-synchronized evaluation
protocol; historical E2 validation timing is not directly comparable to
the new E3_1 baseline. Rank-local epoch accuracy is meaningful, but an
aggregate across ranks at equal epoch number is not a simultaneous global
snapshot. Use wall-clock-aligned interpolation or rank-local curves and label
the evaluation scope correctly.

The Gossip asynchronous ring still uses a Gloo send/receive thread and a
bounded queue. A deferred outbound update can remain queued while optimizer
steps continue; untransmitted updates should not count as completed bytes.
Asynchronous worker shutdown must be validated during a shaped-network pilot.

## Critical scientific boundary

E3_1 **changes communication timing only**. In all cases, model, training data,
worker count, optimizer and communication frequency remain as in E2.
- DDP: a shared all-worker contact gate is called before backward gradient sync.
  It does **not** allow training during a missing global collective; PyTorch
  DDP may still use communication in the forward path.
- DiLoCo H=8 and H=64: an all-worker contact gate at the outer synchronization
  boundary. Outer LR=0.3 and outer momentum=0.9 are preserved; **no extra local
  optimizer updates during a deferred outer sync**.
- Gossip H=8 and H=64: sender thread defers sends until the next predicted
  sender-to-right-neighbor contact window. The original bounded queue is kept,
  so dropped/unsent snapshots can occur. Pre/post-validation NCCL barriers are
  also gated using full all-worker connectivity.
- Flower FL-8 / FL-64: client fit callbacks coordinate a contact-aware
  full-round upload gate. The Flower server and client 0 are colocated;
  local gRPC traffic bypasses link shaping. The gate is before callback
  return, **not an instrumented gRPC ACK**. Downlink/validation phases are
  not yet scheduled independently.

This is a conservative proof-of-concept, not an optimal topology-aware collective
scheduler. For DDP, waiting for every directed pair may be stricter than NCCL's
actual communication graph. All contact predictions are treated as known exactly;
later experiments should perturb the plan to test prediction errors.

## Reservation, setup, and initial pilot

On `fnancy`:

```bash
cd ~/experiments_grid/contact_aware_sched/synthesized_E3_1
./reservation.sh
```

The exact-host request may need adjusting to the current OAR properties/queue.
It must allocate **graffiti-5, graffiti-7, graffiti-8, graffiti-9**.
`reservation_e2.sh` retains the original E2 queue/site logic if you explicitly
need another cluster.

On a **new deploy reservation**, carefully verify the NIC and deployment
image before running `E2_CONFIRM_NETWORK=1 ./setup.sh`. The copied E2
setup performs deployment and network changes; it is not a harmless check.

If E2 was already configured **inside this same OAR reservation**, reuse
existing shared assets instead of redeploying:

```bash
./reuse_e2_runtime.sh
```

Both cases require a working `.venv/bin/python`, CIFAR-100 under `data/`, and
`kavlan_state.json` validated for the current job.

First run the pure scheduler unit tests:

```bash
python3 -m unittest discover -s tests -p 'test_e3_*.py' -v
```

Then verify the setup and run one short **actual-shaping** pilot. Before setting
the following flags, read `NETWORK_MODES.md` and verify the data interface
and safe management/cleanup path. On shared NICs, tc also changes the
management interface's root queue:

```bash
CASE=off DATASET=synthetic MAX_SECONDS=180 \
  TC_APPLY=1 E2_CONFIRM_SHAPING=1 ./gossip/run.sh
CASE=aware DATASET=synthetic MAX_SECONDS=180 \
  TC_APPLY=1 E2_CONFIRM_SHAPING=1 ./gossip/run.sh
```

For `selective_tc` shared management NIC, additionally supply
`E2_CONFIRM_SHARED_TC=1` **only after** validating SSH/recovery.
`TC_APPLY=0` performs a no-shaping diagnostic, not a connectivity experiment.

## Seven E3_1 comparisons

```bash
# One full run per 3h+ allocation; repeat off/aware under equivalent conditions.
CASE=off   DATASET=cifar100 MAX_SECONDS=10800 ./ddp/run.sh
CASE=aware DATASET=cifar100 MAX_SECONDS=10800 ./ddp/run.sh

METHOD=diloco8  CASE=off   ./diloco/run.sh
METHOD=diloco8  CASE=aware ./diloco/run.sh
METHOD=diloco64 CASE=off   ./diloco/run.sh
METHOD=diloco64 CASE=aware ./diloco/run.sh

METHOD=gossip8  CASE=off   ./gossip/run.sh
METHOD=gossip8  CASE=aware ./gossip/run.sh
METHOD=gossip64 CASE=off   ./gossip/run.sh
METHOD=gossip64 CASE=aware ./gossip/run.sh

METHOD=fl8  CASE=off   ./flower/run.sh
METHOD=fl8  CASE=aware ./flower/run.sh
METHOD=fl64 CASE=off   ./flower/run.sh
METHOD=fl64 CASE=aware ./flower/run.sh
```

`CASE=both MAX_SECONDS=180 ./gossip/run.sh` runs both **sequentially**
under the same allocation; when both result summaries exist, `run_tech.sh` automatically generates paired comparison figures. A 3-hour pair cannot fit a 4-hour OAR job;
for full experiments use different reservations, or explicitly acknowledge a
sufficiently long allocation with `E3_CONFIRM_LONG_PAIR=1`.

All runs must use the *same original trace* and network mode.
`E3_MARGIN_S` (default 15 s) is the pre-contact deadline safety margin;
`E3_ESTIMATE_FACTOR` (default 2) scales a simple payload/rate estimate.
The `E3_RENDEZVOUS_LEAD_S` (default .2 s) and rendezvous overhead are
reported, not assumed free.

## Raw and analyzed measurements

E2 files (steps, validation, GPU/NIC samples, TCP socket monitoring,
communication events, trace application logs) are preserved verbatim.

Added:
- `e3_scheduler_decisions_rank*.csv`: **every scheduling decision**, future
  contact start/end, estimated communication duration, safety margin,
  current and selected H (fixed in C1), ready/start/complete/apply timestamps,
  created model version, message and operation IDs, explicit waiting,
  rendezvous overhead, and success/failure state.
- `e3_communication_operations.csv`: normalized operation lifecycle,
  inferred exposure to a planned outage, deferral, ready-to-complete,
  application timing, and logical bytes.
- `e3_outage_progress.csv`: local optimizer steps during the **union** of
  directed outage intervals for each worker (no double counting).
- `e3_recovery.csv`: first completed rank-level communication after outage
  end, or null if not observed.
- `e3_message_staleness.csv`: gossip snapshot age (seconds and receiver/sender
  step difference) at model application, based on matched sender metadata.
- `e3_effective_H.csv`: observed local step interval between successful
  synchronization events, useful for later C2 variable-H experiments.
- `e3_network_counters.csv`, `e3_gpu_outage_idle.csv`: qualified NIC and
  GPU idle proxies, not direct causal or flow-specific measurements.
- `e3_summary.json`: C1/C2 metrics, time-to-accuracy at 50/60/70%,
  successful payload bytes and bytes-to-accuracy, timeout/error events,
  replay fingerprint and termination status.
- `e3_plots/`: standalone accuracy/time and communication timeline.
- `compare_pair.py`: comparison CSV, JSON, accuracy/time, operation-status,
  explicit waiting and delivered-staleness figures.

**Interpretation rules:**
- `exposed_to_scheduled_outage` means an operation was **at risk** because
  its observed lifetime overlapped a predicted outage; it is **not** proof
  of a failed collective. Confirm timeout failures from actual error logs.
- `ready_to_complete_s` includes intentional waiting, computation and
  framework delays. It is not pure network blocked time.
- DDP/DiLoCo logical bytes are input payload, not NCCL on-wire bytes.
  Per-rank NIC counters also cover unrelated traffic.
  `observed_tcp_bytes` and `useful_contact_utilization` remain explicitly
  null if traffic attribution is not verified. The E2 socket-monitor outputs
  remain available for a future validated flow-specific estimator.
- `message_age_s` is computed for delivered Gossip snapshots; undelivered
  and skipped updates must be counted separately.
- Flower `comm_complete_unix` at the next callback is an upper bound on
  previous-upload completion, not a measured gRPC transfer completion.
- Actual applied network transitions are recorded by E2's `replay_rank*.csv`;
  the C1 risk join uses scheduled windows. Validate transition lag before
  claiming that predicted timings matched applied outages.

## Comparison figures

When both runs for the **same technique** are available:

```bash
python3 compare_pair.py \
  --off results/E3_1_gossip64_off_<timestamp> \
  --aware results/E3_1_gossip64_aware_<timestamp> \
  --out comparisons/gossip64_run1
```

The script checks matching method, trace SHA, dataset, optimizer, random seed,
model and duration before plotting. Repeat with different seeds/traces after
the prototype works and estimate paired confidence intervals.

## Status and limitations

This code is committed on a feature branch and is **not yet validated on
Grid'5000 GPUs**. The automated GitHub Actions check compiles the Python code and exercises
scheduler and metrics unit tests, but cannot validate GPU, NCCL, Gloo,
Flower or traffic-control behavior. Before full experiments, run offline tests, a no-shaping
pilot, then a shaped 3-minute run and inspect every `train_rank*.log`,
`replay_rank*.csv`, and `execution_status.json`.

E3_2 will change H; E3_3 will select workers/peers. Both should preserve
the same measurement schema.
