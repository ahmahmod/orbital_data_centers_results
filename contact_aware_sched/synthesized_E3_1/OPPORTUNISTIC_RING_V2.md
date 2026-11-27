# E3_1 opportunistic TCP ring v2

The v1 branch is retained unchanged. v2 keeps the four-worker logical ring and
exact synchronous DiLoCo all-reduce; H, optimizer, dataset split, seed, model,
and 100-epoch cap are unchanged. No topology adaptation or Gossip redesign is
included in this change.

## Transport

Both OFF and AWARE use ordered TCP frames and per-frame application ACKs with
bounded outstanding credits. OFF uses up to E3_RING_MAX_INFLIGHT (default 32).
AWARE limits outstanding + prospective frames by a conservative estimate:
two bandwidth-delay products, serialization of pending/new bytes and ACKs,
one RTT, safety factor and margin, current forward/reverse contact intersection,
and maximum credits. AWARE refills only after ACK frees a credit or contact
opens; it drains already admitted frames even at closure. Prediction is not
a guarantee; late ACKs are counted. No automatic reconnect or resume across
broken TCP connections. This is frame pipelining inside existing chunk/round
dependencies, not fragment-level forwarding across rounds.

## Validation and shutdown

E3_EVAL_STEPS=128 is a multiple of H=8 and H=64. Both synchronous trainers
evaluate the initial model before readiness, record it at trace time zero,
then evaluate at exact local optimizer step checkpoints. NCCL AWARE batches
are capped at the next checkpoint. The initial actual evaluation timestamp
is retained separately; final validation is attempted on normal trainer
completion, including wall-budget stop. A post-horizon validation is labeled
and excluded from in-horizon figures/accuracy targets. Aggregation requires
the same step and four ranks; its timestamp is the latest rank observation.
Epoch progress is step / batches-per-epoch. Final local model can have a
pending unaggregated DiLoCo update; outer_round records the last applied round.

Runner terminates trainer process groups before replay agents, in parallel
across nodes, verifies run-scoped process markers and checks both qdisc and
ownership markers. Normal trainer completion markers distinguish completed
epochs from horizon stop. No blind qdisc deletion. A transient SSH stop error
is a warning only when later remote verification succeeds; cleanup failures
remain failures. Kernel, NFS and SSH behavior require Grid'5000 validation.

## Partial connectivity

Per-TX-chunk counters sum acknowledged application frame payload bytes whose
launch and observed ACK are within the horizon, including frames in incomplete
chunks. Full/partial/none refer to predicted bidirectional availability of
all four logical ring edges. Strict contained bytes have launch and ACK
within the same contiguous connectivity-state interval. Crossing bytes are
reported separately. Partial ACK bytes count ACK observations during partial
connectivity, even if launch occurred in another state. These metrics are not
wire-byte or retransmission measurements. Trace intervals are exported only
if the trace hash matches. Post-horizon drain excluded.

## Run

Prepare a four-GPU OAR reservation as before, then:
```bash
python3 -m unittest discover -s tests -p test_decentralized_ring.py -v
AUTO_DRY_RUN=1 ./auto.sh
# Four ring cases, prioritized by H; 12 hours maximum training.
AUTO_TRANSPORTS=ring E2_CONFIRM_SHARED_TC=1 ./auto.sh
# All eight DiLoCo cases, including native NCCL references.
E2_CONFIRM_SHARED_TC=1 ./auto.sh
```

Each run has a three-hour or 100-epoch budget. Matching v2/v5 results are
skipped; older v1/v4 runs never satisfy the new campaign. Set
AUTO_RETRY_FAILED=1 to rerun failed matching results without overwriting them.
Use AUTO_REPETITION=R02 for an independent repetition. A dry run checks the
planned matrix without launching workers. Results and logs stay in the
existing directory layout with versioned labels.

Recommended pilot: compare OFF/AWARE with 900 seconds each before a full
campaign; inspect peak credits, late ACKs, partial bytes, validation steps,
terminal markers, and cleanup. A continuous-connectivity control and
fixed-window ablation (1, 4, 8, 16, 32 credits) are recommended to separate
pipelining from contact awareness. Repeat paired experiments with identical
hosts and traces before claiming general performance improvements.
