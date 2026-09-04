# System Identification

Everything that turns the real SpiRob into numbers in the MuJoCo model.

## TL;DR

Two independent routes to the same parameters, plus the tooling to record the
data they need:

| Folder | Question it answers | Needs hardware? |
|---|---|---|
| [`direct/`](direct/) | What is *this joint's* stiffness and damping? | no (data is shipped) |
| [`simulation_based/`](simulation_based/) | Which parameters make the *whole robot* move like the real one? | no (data is shipped) |
| [`acquisition/`](acquisition/) | How was the data recorded in the first place? | **yes** |
| [`figures/`](figures/) | Turn results into publication figures | no |
| [`tools/`](tools/) | Poke at the identified model interactively | no |

The direct measurements feed the simulation-based fit as *seeds*. Run them in
that order.

## The pipeline

```
  hardware                     data/                        build/
  ─────────                    ─────                        ──────
  accelerometer  ──record──▶  free_vibration/  ──┐
  ring-down                    joint_NN/*.csv     │
                                                  ├──▶ real2sim.py ──▶ identified
  robot arm      ──manual──▶  static_load/        │      (CMA-ES)       parameters
  + scale                      joint_NN.csv    ───┘                     + model XML
                                                  │
  GoPro + ArUco  ──track───▶  trajectories/    ───┘
  + motor forces               *.parquet
```

1. **`direct/free_vibration_gui.py`** — the authoritative per-joint `k` and `d`.
   Writes `results.k_mean` / `results.d_mean` into each
   `data/free_vibration/joint_NN/sysid_settings.yaml`.
2. **`simulation_based/real2sim.py --mode validate`** — seeds the model from
   those YAMLs and reports the baseline cost. **Always run this first.**
3. **`simulation_based/real2sim.py --mode finetune`** — optimises around the
   seeds.

## Quick start

```bash
# batch-evaluate every free-vibration recording           (~1 min)
uv run sysid/direct/free_vibration.py

# static load test, all four joints                       (~2 s)
uv run sysid/direct/static_load.py

# baseline: identified seeds, one forward simulation      (~1 min)
uv run sysid/simulation_based/real2sim.py --mode validate

# fit: CMA-ES around those seeds                          (hours)
uv run sysid/simulation_based/real2sim.py --mode finetune \
    --optimizer cma --maxiter 500 --workers 8

# the validation figure from the result                   (~1 min)
uv run sysid/figures/fig_validation.py
```

Ctrl+C during a finetune stops cleanly and still saves the best result so far.

## Joint numbering — read this before touching anything

There is **one** convention and everything follows it:

* **Model index `i`** = MuJoCo joint index. The model tree runs **from the
  base**, so index `0` is the base joint (named `j_12`) and index `12` is the
  tip (`j_0`). The XML name counts *down*; the index counts *up*. This is
  confusing exactly once.
* **Real joint `N`** in the datasets is `joint_N_deg`, and `joint_1` is the
  **base**. So real `joint_N` maps to model index `N−1`.
* Every per-joint array — seeds, `qpos`, the stiffness/damping/frictionloss
  vectors, the optimiser output — is in **model index order**.
* `joint_index_labels(model, reverse)` in `real2sim.py` is the single source of
  truth, and the result JSON carries an explicit `joints` list pairing
  `model_index` / `model_joint` / `real_joint` so nothing has to be inferred.
* `--reverse-real-joints` exists only for a dataset that numbers joints from
  the tip. The shipped data does not.

## What is *not* here

Superseded scripts were dropped rather than carried over. The following existed
in the original working repository and are intentionally absent:

| Dropped | Superseded by |
|---|---|
| `sysid_real.py`, `sysid_real_alt.py`, `sysid_real_alt_plot.py`, `sysid_real_linear_profile.py` | `simulation_based/real2sim.py` |
| `sysid_simple.py`, `spirob_sysid.py` | `simulation_based/sim2sim.py` |
| `cv_aruco_test.py` | `acquisition/track_and_sync.py` (tracking + sync in one pass) |
| `sys_id_record.py` | `acquisition/record_trajectory.py` (scripted force profiles) |
| `spirob_sysid_viewer.py`, `sysid_real_visualize.py` | `tools/live_dashboard.py` |
