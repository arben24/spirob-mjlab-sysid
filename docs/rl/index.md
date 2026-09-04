# Reinforcement learning

## TL;DR

The second half of this repository trains tendon-force policies for the SpiRob
in [mjlab](https://github.com/mujocolab/mjlab) — Isaac-Lab's manager-based API
on top of GPU-accelerated MuJoCo-Warp.

* **Four goals on one tentacle**: [reach, shape, trajectory,
  wrap](tasks.md), each crossed with a five-rung **sensor-ablation ladder**.
  The actor sees only what the real rig can measure; the critic always sees the
  full state, so a comparison across levels isolates the sensor suite.
* **Domain randomisation is on by default**, phased in by a curriculum. That is
  the direct consequence of the identification result: the fitted model is not
  right, so a policy must not be trained as if it were.
* The RL half lives in its own uv project (`rl/`, Python 3.13) because mjlab
  brings torch, CUDA and MuJoCo-Warp; the identification half stays on 3.10 and
  never sees them. See [Training](training.md).
* A trained policy can leave simulation: [the rig bridge](rig.md) rebuilds the
  exact observation vector from live sensor frames and closes the loop over
  serial.

```bash
cd rl
uv sync
uv run train RlExplor-Spirob-Tcp-Reach --env.scene.num-envs 4096
```

## How this follows from the identification

The identification chapter ends on a negative result: the three methods
disagree, the fitted parameters are physically implausible, and the residual
gap is [structural, not an optimisation problem](../sysid/results.md). The
rigid-chain-plus-torsion-spring model has no term for tendon friction in the
guide rings, TPU non-linearity or joint coupling, so the optimiser compensates
with numbers that minimise cost and mean nothing physically.

The practical conclusion is not "identify harder". It is:

1. Calibrate as far as the model structure allows — **done**, that is `sysid/`.
2. Treat the result as the *centre of a distribution*, not as the truth, and
   train across that distribution.

So the calibrated numbers enter here as the nominal XML values, and every
dynamic parameter that the identification could not pin down is randomised
around them in proportion to how poorly it is known. Under domain randomisation
the individual number matters much less — which is exactly why the
implausibility of the identified set is tolerable.

## What exists

| Piece | Where |
|---|---|
| Task family (reach / shape / trajectory / wrap × 5 sensor levels) | `rl/src/spirob_rl/tasks/spirob/` |
| Training / playback / inference entrypoints | `rl/src/spirob_rl/{train,play,run,infer}.py` |
| The model the policies train on | `models/spirob_13seg_rl.xml` |
| Measured table of statically holdable poses (shape task targets) | `rl/src/spirob_rl/tasks/spirob/holdable_poses.npz` |
| Hardware bridge, target GUI, telemetry | `rl/src/spirob_rl/rig/` |
| Reachability map of a trained reach policy | `rl/src/spirob_rl/rig/workspace_sweep.py` |

## The interface between the two halves

`sysid/` produces a model; `rl/` consumes one. Nothing in `sysid/` imports from
`rl/` and nothing in `rl/` imports from `sysid/` — the handover is a tracked
model XML and a parameter JSON, never a Python import. Both may use the shared
library in `src/spirob/`, and both resolve every path through `spirob.paths`:
the tasks load `RL_MODEL` from `models/`, and everything the RL scripts generate
lands under `build/rl/`.

## Where to go next

* [Tasks](tasks.md) — the four goals, the sensor ladder, rewards and commands.
* [Training](training.md) — environment setup, running a training, what the
  knobs do, domain randomisation.
* [Rig bridge](rig.md) — running a trained policy on the real robot, and the
  workspace map.
