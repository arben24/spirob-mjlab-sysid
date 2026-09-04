# Reinforcement Learning

> **Status: scaffolded, not implemented.** This folder holds the brief, the
> interface contract and the empty structure. No training code exists yet.

## TL;DR for whoever builds this

The system-identification half of this repository ([`sysid/`](../sysid/)) is
finished, and its conclusion determines how the RL half must be built:

**Do not train against a single calibrated model.** The identified parameters
reproduce the real robot's gross motion (6.8° mean joint RMSE) but not its
trajectory, and they are physically implausible — the optimiser compensates for
missing model terms (tendon friction in the guide rings, TPU non-linearity,
joint coupling) with numbers that minimise cost and mean nothing. The residual
sim-to-real gap is **structural**, not a matter of better fitting.

The intended strategy is therefore two-stage:

1. Calibrate the model as far as the current structure allows — done, see
   [`data/identified/real2sim_cma_500iter.json`](../data/identified/).
2. Bridge the rest with **domain randomisation**: train across a *range* of
   model variants centred on those values, not on one exact set.

Under domain randomisation the individual number matters much less, which is
precisely why the implausibility of the identified set is tolerable.

## What you are given

| Asset | Path |
|---|---|
| Identified model, ready to load | `models/spirob_13seg_identified.xml` |
| Nominal model | `models/spirob_13seg.xml` |
| Parameter set as JSON (per joint, per tendon, solver knobs) | `data/identified/real2sim_cma_500iter.json` |
| Parametric model generator | `spirob.generate_xml_string(...)` |
| Rollout loop, controllers, contact forces | `spirob.simulate` |
| A real 60 s trajectory to sanity-check against | `data/trajectories/*.parquet` |

## Interface contract

Keep `rl/` and `sysid/` independent. `sysid/` produces a model; `rl/` consumes
one. Nothing in `sysid/` may import from `rl/`, and `rl/` should reach into
`sysid/` only through the artefacts above — a model XML and a parameter JSON —
never by importing an identification script.

Both may share `src/spirob/`.

## Model facts you need

* **13 hinge joints**, model index 0 = base (`j_12`), index 12 = tip (`j_0`).
  Every per-joint array in this repo is in model index order.
* **2 tendons**, 2 force actuators, `ctrlrange = [-150, 0]` — **pull only**.
  Positive control values are silently no-ops, which is a very easy sign bug to
  ship. Assert it.
* Joint limits: ±24.45° per joint.
* Timestep 0.004 s, Newton solver, elliptic cone, `impratio = 15`.
* Measured tendon forces reach ~110 N, so an action space must cover that.

## Suggested randomisation ranges

Centred on the identified values, widened by what the identification actually
established:

| Parameter | Centre | Suggested range | Why |
|---|---|---|---|
| joint stiffness | per-joint, 0.37–0.83 | ×[0.5, 2.0] | the *least* identifiable quantity — 28–158 % error even in the best case |
| joint damping | per-joint, 2e-4…2e-3 | ×[0.3, 3.0] | 4–31 % error sim-to-sim |
| tendon stiffness | from XML | ×[0.9, 1.1] | recovered to <5 %, so randomise it least |
| joint frictionloss | 0.15 (unmeasured) | [0, 0.4] | never measured at all — treat as fully uncertain |
| segment mass | from geometry | ×[0.9, 1.1] | 3D-print infill varies |

Rationale: randomise each parameter in proportion to how poorly it is known.
See [`docs/sysid/results.md`](../docs/sysid/results.md) for where those error
figures come from.

## Planned layout

```
rl/
├── envs/       environment definitions (observation, action, reward, termination)
├── configs/    training and randomisation configs
└── scripts/    train / evaluate / export entry points
```

The repository is named `spirob-mjlab-sysid` because
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp based, Isaac-Lab-style)
is the intended training stack. That choice is not yet locked in — nothing here
depends on it.

## Open questions

* **Task.** Reaching a target with the tip? Grasping? Following a shape?
* **Observation.** Joint angles only (matching what ArUco can measure), or
  privileged state in simulation with an asymmetric critic?
* **Action.** Direct tendon forces, or a delta on top of a baseline controller?
* **Real-robot loop.** Only the two motor forces and rope lengths are available
  live; joint angles need the camera. That constrains any deployable policy's
  observation space.
