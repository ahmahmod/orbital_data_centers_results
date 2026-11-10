# E1_a → new_E1_b reporting alignment (patch)

Target repository folder: `synthesized_contact_plan/E1_a/` (not `new_E1_b/`).

## Included files
- `train.py` — preserves DDP training, adds identical-holdout per-rank evaluation at epoch boundaries; writes `validation_rank{rank}.csv` and `validation_aggregate.csv`, while retaining rank-0 `validation.csv` and checkpoints.
- `run.sh` — invokes the three additional postprocessing scripts at the end of a completed run.
- `communication_events.py` — normalizes traced DDP gradient bucket completion events into `communication_events.csv`; destinations intentionally empty because AllReduce is a collective, not a unicast.
- `plot_comparison.py` — generates E1_b-style plot names 20–24 and compatibility plots 08 and 10, alongside the existing E1_a figures; creates `plot_summary.json`.
- `comparison_metrics.py` — creates `comparison_metrics.json`, with explicit logical-versus-TCP measurement semantics.

## Apply
```bash
cd ~/experiments_grid/synthesized_contact_plan
cp -a E1_a E1_a_backup_$(date +%Y%m%d_%H%M%S)
unzip -o E1_a_alignment_patch.zip -d .
cd E1_a
bash -n run.sh
../.venv311/bin/python -m py_compile train.py communication_events.py plot_comparison.py comparison_metrics.py
N_NODES=4 GPUS_PER_NODE=1 MEASURE_TCP=1 MAX_SECONDS=240 ./run.sh
```

The ZIP contains `E1_a/` as its top folder. It does **not** include the existing training data, environments or results.

## Important experiment interpretation
- All four DDP replicas independently evaluate the *same* validation holdout; this adds GPU computation and changes validation-phase wall time relative to prior E1_a runs. Do not compare total wall time to historical runs without controlling for this.
- `communication_events.csv` counts traced input bucket payload once per submitting rank, **not transferred TCP bytes**, and not a src→dst transfer.
- If TCP rank attribution is missing, `comparison_metrics.json` returns `null`, **not zero**.
- `backward_with_comm_s` in E1_a includes overlapping DDP AllReduce, while E1_b local backward does not. Plots preserve this distinction.
- E1_b has method-specific DiLoCo and gossip events that are not applicable to DDP. Do not fabricate `messages_rank*.csv` in E1_a.
- This package passed compilation, bash syntax, and synthetic postprocessing tests. Four-GPU execution has not been tested here.
