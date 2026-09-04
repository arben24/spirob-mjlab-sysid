# Reinforcement learning

!!! warning "Not implemented"
    This half of the repository is **scaffolded, not built**. The folder
    structure, the brief and the interface contract exist; no training code
    does.

## TL;DR

The system-identification half is finished, and its conclusion dictates how this
half must be built:

**Do not train against a single calibrated model.** The identified parameters
reproduce the real robot's gross motion (6.8° mean joint RMSE) but not its
trajectory, and they are physically implausible. The residual gap is
[structural, not a fitting problem](../sysid/results.md).

The intended strategy is two-stage:

1. Calibrate as far as the model structure allows — **done**.
2. Bridge the rest with **domain randomisation**: train across a range of model
   variants centred on the identified values.

Under domain randomisation the individual number matters much less, which is
exactly why the implausibility of the identified set is tolerable.

## What is available

| Asset | Path |
|---|---|
| Identified model | `models/spirob_13seg_identified.xml` |
| Nominal model | `models/spirob_13seg.xml` |
| Parameters as JSON | `data/identified/real2sim_cma_500iter.json` |
| Parametric generator | `spirob.generate_xml_string(...)` |
| Rollout loop, controllers, contact forces | `spirob.simulate` |
| A real 60 s trajectory to check against | `data/trajectories/*.parquet` |

## Suggested randomisation ranges

Randomise each parameter in proportion to how poorly the identification pinned
it down:

| Parameter | Centre | Range | Why |
|---|---|---|---|
| joint stiffness | per-joint, 0.37–0.83 | ×[0.5, 2.0] | least identifiable — 28–158 % error even in the best case |
| joint damping | per-joint, 2e-4…2e-3 | ×[0.3, 3.0] | 4–31 % error sim-to-sim |
| tendon stiffness | from XML | ×[0.9, 1.1] | recovered to <5 %, so randomise least |
| joint frictionloss | 0.15 (unmeasured) | [0, 0.4] | never measured — treat as fully uncertain |
| segment mass | from geometry | ×[0.9, 1.1] | 3D-print infill varies |

## Model facts

* **13 hinge joints**, model index 0 = base (`j_12`), index 12 = tip (`j_0`).
* **2 tendons**, 2 force actuators, `ctrlrange = [-150, 0]` — **pull only**.
  A positive control value is a silent no-op. Assert it.
* Joint limits ±24.45°, timestep 0.004 s, Newton solver, elliptic cone,
  `impratio = 15`.
* Measured tendon forces reach ~110 N.

## Interface contract

`sysid/` produces a model; `rl/` consumes one. No imports across that boundary —
`rl/` reaches into the identification only through the artefacts above (a model
XML and a parameter JSON), never by importing an identification script. Both may
share `src/spirob/`.

## Open questions

* **Task** — tip reaching? grasping? shape following?
* **Observation** — joint angles only (what ArUco can actually measure), or
  privileged simulator state with an asymmetric critic?
* **Action** — direct tendon forces, or a delta on a baseline controller?
* **Real-robot loop** — only the two motor forces and rope lengths are available
  live; joint angles need the camera. That constrains any deployable policy.

Full brief: [`rl/README.md`](https://github.com/arben24/spirob-mjlab-sysid/blob/main/rl/README.md).
