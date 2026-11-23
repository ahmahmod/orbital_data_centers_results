# Synthetic satellite contact plan for 2- and 4-worker distributed training

## Files
- `directed_trace_2workers.csv`: 26 directed intervals for W1 ↔ W2; **immediately compatible with the existing 2-node `replay_grid5000.py`**.
- `directed_trace_4workers.csv`: 122 directed intervals for the six unordered edges; **requires per-destination traffic shaping or a topology emulator for 4 workers**.
- `generate_demo.py`: dependency-free deterministic generator for both traces.
- `plot_plan.py`: plot generator, requiring `matplotlib`.
- `phase_guide.csv`: suggested experiments.
- `manifest.json`: assumptions and limitations.
- `contact_plan_overview.png`, `four_worker_links.png`: visualized plans.

## Run two-worker test
From a Grid'5000 frontend with a configured two-node isolated data VLAN and verified experimental interfaces:

```bash
python3 replay_grid5000.py \
  --trace directed_trace_2workers.csv \
  --src SAT-DEMO-001 --dst SAT-DEMO-002 \
  --node-a "$NODE_A" --node-b "$NODE_B" \
  --iface-a "$IFACE_A" --iface-b "$IFACE_B" \
  --rate-scale 1 --log results/ddp_demo_replay.jsonl
```

Omit `--apply` for validation. Add `--apply` only after confirming the interface and management isolation. Run your DDP job concurrently under a shared clock; *starting DDP and network replay manually at different times does not ensure timestamp alignment*. For the first network-only check, you may use the existing TCP runner with `--trace directed_trace_2workers.csv`, `--src SAT-DEMO-001`, and `--dst SAT-DEMO-002`.

## Phases of W1–W2, using half-open time intervals
| Seconds | Mbps | Interpretation |
|---|---:|---|
| 0–360 |100| Stable control / DDP initialization |
| 360–540 |20| Bandwidth bottleneck; adapt accumulation |
| 540–660 |0| 2-min outage |
| 660–1080 |100| Recovery |
| 1080–1260 |30| Degradation |
| 1260–1440 |0| 3-min outage |
| 1440–1980 |120| Long contact |
| 1980–2160 |15| Strongly constrained |
| 2160–2340 |0| 3-min outage |
| 2340–2880 |80| Reconnected |
| 2880–3060 |25| Degradation |
| 3060–3180 |0| 2-min outage |
| 3180–3600 |100| Final contact |

## IMPORTANT
- **Synthetic**: The rates, intervals, delay parameters, and worker assignment do not come from SpaceX spacecraft or any orbit propagation. Do not publish as actual SpaceX contact-plan output.
- The two-worker trace contains symmetric directional data and explicit outages across the entire hour.
- `--rate-scale 1` retains the illustrative Mbps speeds. In contrast to the original 10-Gbps generated traces, multiplying by `0.01` here would produce 1 Mbps for a 100 Mbps interval.
- A standard PyTorch DDP group cannot simply omit W3 mid-allreduce. Worker selection requires epoch/round-boundary reconfiguration, TorchElastic-style restarts, or a different synchronization approach.
- `no_sync()` can suppress gradient communication for local backward passes, but optimizer semantics must be defined and identical across workers. Long outages may require sufficient accumulation storage or planned pauses.
- The **four-worker** trace defines per-pair conditions; assigning the same root `netem` to every NIC would not reproduce them. Use `tc` per-destination filters/classes, network namespaces/virtual routers, or topology emulation before drawing conclusions.
- All workers in a given DDP collective must have compatible collective calls. Keep all control/rendezvous traffic separate from the emulated data interfaces; otherwise network connectivity can be unintentionally bypassed.
