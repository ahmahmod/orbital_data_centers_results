# E3_1 decentralized contact-aware synchronous ring

This experimental branch defaults to a custom four-worker, point-to-point
TCP ring for DDP, DiLoCo8 and DiLoCo64. The existing native NCCL path remains
available with E3_SYNC_TRANSPORT=nccl. Gossip/Flower remain unchanged.

## Architecture

- Train/main thread: forward, backward, optimizer, and the synchronous
  reduce-scatter/all-gather dependency state machine.
- TX thread: owns a persistent TCP connection to (rank+1)%4. It schedules
  individual frames using only the outgoing contact and its reverse ACK path.
- RX thread: owns the connection from (rank-1)%4. It uses the incoming
  contact plan to arm reception and always drains already-buffered/in-flight
  traffic, even after predicted closure, to avoid withholding acknowledgments.
- No frontend broker, shared-filesystem grant/ACK polling, global contact
  intersection or central scheduler is used during ring training.
- Gloo synchronizes the initial model before traffic shaping and is destroyed
  before training. The runner still coordinates startup and tc cleanup once.
- Every frame contains collective step, phase, round, chunk, byte offset and
  length. The receiver validates sequence and sends an application ACK.
  At most one frame is outstanding per outgoing link; queues are bounded.
  TCP retransmits within a live connection. A broken connection or invalid
  frame fails the run; it does not reconnect or reapply a partial update.

For four ranks, an exact mean is computed by three reduce-scatter rounds and
three all-gather rounds. Local send/receive dependencies constrain progress,
but all ring links do not need to be open simultaneously. The ring order
is fixed 0->1->2->3->0. Each TCP data direction also needs its reverse ACK
direction. No adaptive membership, quorum, stale gradients, or extra local
optimizer steps are introduced.

DDP-equivalent training: H=1, flat scaled FP32 gradients are averaged,
copied back to GPU, then GradScaler/SGD apply the global update. This is
not the native PyTorch DDP reducer and does not preserve bucket overlap.
DiLoCo: H=8 or 64, the original model-delta sign, outer momentum and outer
learning rate are retained. Both use the existing ResNet18/data/validation
and resource metrics. Local batch-normalization buffers are not averaged.

## OFF versus AWARE

Both custom modes use identical TCP sockets, framing, FP32 aggregation,
GPU/CPU copies and algorithm. OFF submits frames immediately regardless
of the trace and relies on TCP recovery. AWARE checks the contact duration,
data rate, ACK direction, propagation delay, opening guard and safety
margin before launching each frame. An estimated feasible frame may still
overrun a contact due to TC application jitter, congestion or TCP backoff;
the implementation does not cancel an already-issued frame. RX waits on
its local predicted incoming window, but reads buffered bytes immediately.

The stop vote is appended as one scalar to each reduction. If any worker
starts a collective after the experiment deadline, the resulting vote
prevents all workers from applying that terminal update. Pending transfers
without a feasible contact wait until the trace horizon plus a cleanup guard,
then drain outside the measured horizon. One already-started collective may
complete after the horizon before the terminal vote; raw logs retain it, and
E3 summary/comparison metrics censor it. DiLoCo can finish its fixed local
block before the stop vote; those local steps are not an adaptive H policy.
A hard drain timeout and the existing runner grace bound cleanup. Actual
post-horizon drainage assumes tc cleanup succeeds; a failure is not hidden.

## Run on Grid'5000

From the existing prepared four-GPU OAR reservation:

~~~bash
git fetch origin
git switch --track origin/e3-1-decentralized-ring
cd contact_aware_sched/synthesized_E3_1
E2_CONFIRM_SHARED_TC=1 CASE=aware E3_SYNC_TRANSPORT=ring \
DATASET=synthetic E2_LOG_EVERY=1 MAX_SECONDS=900 ./ddp/run.sh
~~~

Repeat with CASE=off and the same trace, settings and frame size. Use the
same commands with METHOD=diloco8 or METHOD=diloco64 and ./diloco/run.sh.
After smoke tests, use DATASET=cifar100 and MAX_SECONDS=10800 as appropriate.
For native reference, run E3_SYNC_TRANSPORT=nccl CASE=off. Do not treat
native NCCL vs custom ring as a pure scheduling ablation. compare_pair.py
enforces matching transport profiles; use separate analysis for native
vs custom overhead.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| E3_SYNC_TRANSPORT | ring | ring or nccl |
| E3_RING_FRAME_BYTES | 1048576 | Payload per frame, multiple of 4 |
| E3_RING_PORT | master port + 1 | TCP listener on each worker IP |
| E3_RING_MARGIN_S | 0.5 | Closing safety margin per frame |
| E3_RING_OPEN_GUARD_S | 0.25 | Wait after contact opens |
| E3_RING_FACTOR | E3_ESTIMATE_FACTOR (2) | Transfer estimate multiplier |
| E3_RING_MIN_TRANSFER_S | 0.02 | Minimum frame time estimate |
| E3_RING_CONNECT_TIMEOUT_S | 180 | Socket startup timeout |
| E3_RING_DRAIN_GUARD_S | 2 | Guard after trace horizon |
| E3_RING_DRAIN_TIMEOUT_S | min(240, shutdown grace - 10) | Hard drain deadline |

A smaller frame fits shorter contacts; a larger one reduces frame ACK/RTT
overhead. The default 1 MiB is experimental, not calibrated. The port must
be reachable between workers and must differ from the Gloo master port.
No frontend TCP control server is started for the ring transport.

## Metrics and diagnostics

Each rank produces e3_ring_transfers_rankN.csv, with direction, peer,
step, phase, round, chunk, timestamps, payload, frame counts, local
contact waiting, I/O time, out-of-window launches, ACK overruns and
post-horizon drain counts. Payload bytes are acknowledged application
payload, not TCP wire bytes. RX time includes waiting for peer data.
The existing progress, scheduler and validation CSV files are retained.
E3 summaries include ring_transfer_metrics and a horizon-censored flag.
Legacy plots may still display raw post-horizon cleanup samples; use
the E3 summary and comparison for fixed-horizon performance conclusions.

## Tests

~~~bash
python3 -m unittest discover -s tests -p test_decentralized_ring.py -v
python3 -m py_compile communication/*.py ring_train.py runner.py \
e3_analyze.py compare_pair.py
~~~

The CPU tests use real loopback sockets, verify exact mean and in-band stop,
and simulate non-overlapping contact windows. They do not emulate packet
loss or validate GPU training speed. Run a four-GPU synthetic experiment
with tc replay before committing full CIFAR-100 reservations.
