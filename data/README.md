# Data

All measured data, 12 MB, tracked in git. Nothing here is generated; nothing
here is written to by any script.

```
data/
├── free_vibration/     ring-down recordings + identified k, d per joint
├── static_load/        mass/angle series of the static load test
├── trajectories/       the 60 s reference recording (forces + ArUco angles)
└── identified/         result JSONs of the identification runs
```

## `free_vibration/joint_NN/`

One folder per measured joint: **`joint_01`** (base), **`joint_08`** (middle),
**`joint_11`**, **`joint_13`** (tip).

| File | Content |
|---|---|
| `spirob_messung_<timestamp>.csv` | one accelerometer ring-down; ~800 Hz, hardware timestamps |
| `sysid_settings.yaml` | **the authoritative result** |

The YAML holds three blocks:

* `settings` — analysis parameters, above all **`J`**, the moment of inertia of
  the swinging segment. It is hand-tuned in the GUI, and `k` and `d` scale
  linearly with it.
* `settings.start_overrides` — the manually chosen start of the oscillation, per
  file. These exist only in the GUI and are why the batch script can differ from
  these results by a few percent.
* `results` — `k_mean`, `d_mean`, `zeta_mean` and the full per-file breakdown.
  **`results.k_mean` / `results.d_mean` are what seed the real-to-sim fit.**

| Joint | `k` [N·m/rad] | `d` [N·m·s/rad] | `ζ` |
|---|---:|---:|---:|
| 1 (base) | 0.670 | 1.91e-3 | 0.125 |
| 8 | 0.832 | 0.96e-3 | 0.133 |
| 11 | 0.370 | 0.26e-3 | 0.129 |
| 13 (tip) | 0.692 | 2.28e-3 | 0.134 |

The folder for the tip joint is named `joint_13` here. In the original working
repository it was `data_joint_14`; the loader clamps anchor numbers into
`[1, njnt]`, so both land on the tip either way.

## `static_load/`

`joint_NN.csv` — the mass on the scale and the joint deflection, per measurement
point. `lever_arms.csv` — the lever arm `r` per joint, needed to turn force into
torque. Comment lines start with `#`.

## `trajectories/`

`spirob_tendon_trajectory_60s.parquet` — the reference recording the real-to-sim
identification is fitted against. 60 s, both tendons driven through scripted
force phases, joint angles from ArUco tracking.

Key columns: `joint_1_deg` … `joint_13_deg` (**joint 1 = base**),
`meas_force_0_N` / `meas_force_1_N` (the simulation's input), `cmd_force_*_N`,
`meas_length_*_mm`, `global_timestamp_s`.

The video these angles were tracked from (~770 MB) is **not** in the repository.

## `identified/`

| File | Produced by |
|---|---|
| `real2sim_cma_500iter.json` | `real2sim.py --mode finetune --optimizer cma` — **the reference parameter set** |
| `real2sim_de_500iter.json` | the same with Differential Evolution, for comparison |
| `sim2sim_de_vs_cma.json` | `sim2sim.py --compare` |
| `sim2sim_sensitivity.json` | `sim2sim.py --sensitivity` |
| `sim2sim_cma.json`, `sim2sim_de.json` | single sim-to-sim runs |
| `free_vibration_summary.csv`, `free_vibration_per_file.csv` | `free_vibration.py` |

A result JSON carries an explicit `joints` list pairing `model_index`,
`model_joint` and `real_joint` with the values, plus flat arrays that index
directly into `model.jnt_stiffness` / `dof_damping` / `dof_frictionloss`. No
ordering has to be inferred.

## Preprocessing

The real data is smoothed with a moving average before optimisation. Without it
the optimiser fits high-frequency measurement noise instead of robot dynamics.
`real2sim.py` writes the before/after comparison per channel to
`build/preprocessing_plots/` on every run — check it if a fit behaves oddly.
