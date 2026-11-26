## Startup synchronization and replay (new)

The DDP coordinator now implements a **two-stage, shared-filesystem startup
barrier**. It launches the four DDP trainers *without applying any tc rules*.
Each rank loads CIFAR-100, initializes NCCL/DDP, completes its initial
process-group barrier, and atomically writes a ready record. The coordinator
waits until all four current-run records are visible; it then selects a
**new** common trace epoch (default 150 seconds in the future), launches the
four tc replay agents, and verifies that each agent has installed its initial
qdiscs/filters (or confirmed dry-run). Only then does it publish the start
timestamp that releases the four training workers.

The default readiness timeout is `E3_STARTUP_READY_TIMEOUT=1800` seconds,
and can be overridden if the GPU deployment is exceptionally slow. An
incomplete startup now fails **before traffic shaping begins**. A failed TC
setup fails before training is released, and cleanup checks still apply.

Training logs include `worker_ready` and `shared_start_received`.
The coordinator prints `DDP AND ALL TC AGENTS READY` only after both
startup phases succeed. The final timestamp is written to
`e2_run_config.json` as `init_unix_s`, with per-rank readiness timestamps
in `readiness_unix_s`.

The step-level scheduler still requires all ranks to rendezvous. A timed-out
rendezvous now reports present/missing ranks and the ages of submitted
readiness files. Failures from peers are propagated through atomic failure
markers instead of waiting for the full 180-second step timeout.

**Caveat:** This handshake guarantees launch ordering and a common
scheduled epoch, not zero clock skew, guaranteed end-to-end contact
completion, or safe cleanup after abrupt SIGKILL. Confirm no residual
`htb 1:` qdisc remains before repeating a run.

# DDP — E3_1 contact-aware step admission

Only the original full contact-aware scheduling policy is supported for DDP.
Use `CASE=aware ./run.sh`, or omit CASE (the DDP launcher defaults to
`aware`). Attempts to use `CASE=off`, `CASE=proactive`, `CASE=both`, or
`CASE=guardpair` are rejected.

For a functional pilot:
```bash
METHOD=ddp CASE=aware DATASET=synthetic MAX_SECONDS=180 TC_APPLY=0 ./ddp/run.sh
```
Run from the `synthesized_E3_1` directory inside an active GPU reservation.

DDP implements **blocking synchronous gradient updates**. The contact-aware
scheduler admits each iteration before model forward and gradient backward.
The DDP instance has `broadcast_buffers=False` to avoid an unplanned
forward buffer synchronization. Initialization/broadcast collectives before
the scheduled start are still not contact-gated.

**Coordination latency:** DDP now uses a zero additional rendezvous lead
(default shared scheduler lead for other methods is 0.2 seconds). The shared
filesystem still performs per-step coordination; its actual cost is logged.
The `E3_RENDEZVOUS_LEAD_S` environment variable can explicitly override
the DDP default for controlled experiments.

**Contact policy:** Remains deliberately conservative: all 12 directed
worker contacts must provide a contiguous safe time budget. We have NOT
measured NCCL's communication graph, so the code does not guess a reduced
ring topology. Investigating the actual NCCL transport and its required
links is a separate optimization requiring verification.

**Measurements:** `e2_events_rank*.csv` includes
`scheduler_gate_finished` (the elapsed wall-clock waiting from the point
the batch is ready until admission) and `scheduler_horizon_stop` when no
safe contact remains. The scheduler decisions CSV separately records
`rendezvous_wait_s`, predicted contact window, and
`defer_duration_s`. A run ending at its contact horizon may not reach
100 epochs; do not mislabel it as a 100-epoch completion.

**Safety:** With `TC_APPLY=1`, the replay changes tc qdiscs. If the
deployment uses `selective_tc` on the management NIC, review the network
safety checks and provide `E2_CONFIRM_SHARED_TC=1` only after verifying
management access and a recovery path.

The fixes have offline unit tests, but they require an actual four-GPU
Grid'5000 pilot to validate NCCL behavior and performance.
