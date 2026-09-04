# Reinforcement Learning

Training tendon-force policies for the SpiRob with
[**mjlab**](https://github.com/mujocolab/mjlab) (Isaac-Lab-style manager API on
top of MuJoCo-Warp), plus the bridge that runs a trained policy on the real rig.

## TL;DR

* Four task variants on one tentacle — **reach**, **shape**, **trajectory**,
  **wrap** — each crossed with a five-rung **sensor-ablation ladder**, so 40
  registered task ids in total (each variant × level also has a `-DrPlay` twin).
* The actor sees only what the rig can measure; the critic always sees the full
  state. Which sensors the actor gets is the experiment.
* **Domain randomisation is on by default**, ramped in by a curriculum. That is
  the direct consequence of the identification result: the model is not right,
  so do not train as if it were (see [`docs/sysid/results.md`](../docs/sysid/results.md)).
* This folder is **its own uv project** (Python 3.13, own `.venv`). The
  identification half stays on 3.10 and never sees torch.

```bash
cd rl
uv sync                                   # ~4 GB: torch, MuJoCo-Warp, mjlab
uv run train RlExplor-Spirob-Tcp-Reach --env.scene.num-envs 4096
uv run play  RlExplor-Spirob-Tcp-Reach --checkpoint-file ../build/rl/logs/<exp>/<run>/model_499.pt
```

## Why a separate environment

mjlab pulls torch, CUDA and MuJoCo-Warp — several GB — and was developed against
Python 3.13, while the identification half is pinned to 3.10 and its CI matrix
runs 3.10/3.12. Two projects means installing mjlab cannot disturb a working
sysid setup, and the RL side runs on the interpreter its stack was built for.

The root package is still a dependency (editable, from `..`): `spirob.paths` is
how the tasks find `models/` and how every script here writes into `build/`.
Nothing here imports from `sysid/`, and nothing in `sysid/` imports from here —
`sysid/` produces a model, `rl/` consumes one.

## Commands

All of these run from `rl/`.

```bash
uv sync                                # create rl/.venv (Python 3.13)
uv sync --extra hardware --extra dev   # + pyserial, pytest, ruff

# Training. Checkpoints land in build/rl/logs/<experiment_name>/<timestamp>/.
uv run train RlExplor-Spirob-Tcp-Reach --env.scene.num-envs 4096
uv run train RlExplor-Spirob-Wrap-Imu --env.scene.num-envs 4096 --agent.logger tensorboard

# Watch a checkpoint in the viewer (add -DrPlay to watch it under full DR).
uv run play RlExplor-Spirob-Tcp-Reach --checkpoint-file <path>/model_499.pt
uv run play RlExplor-Spirob-Tcp-Reach-DrPlay --checkpoint-file <path>/model_499.pt

# Run a checkpoint on observations from a file or stdin (what the rig bridge uses).
uv run infer RlExplor-Spirob-Tcp-Reach --obs-file obs.npy

# Reachability map: drive a trained reach policy across a grid of targets.
uv run python -m spirob_rl.rig.workspace_sweep --figure
uv run python -m spirob_rl.rig.workspace_figure --sweep ../build/rl/workspace/sweep_*.npz --threshold 30

# On the real rig (needs the hardware extra and the ESP32 firmware running).
uv run python -m spirob_rl.rig.policy_bridge RlExplor-Spirob-Tcp-Reach-Imu \
    --port /dev/ttyUSB0 --joint-port /dev/ttyUSB1 --dry-run
uv run python -m spirob_rl.rig.target_gui     # commanded target vs. measured tip

uv run --extra dev pytest              # task registration + env construction (needs a GPU)
```

`--agent.logger tensorboard` is worth knowing: mjlab logs to **wandb** by
default and will ask to log in on the first run.

## Layout

```
rl/
├── pyproject.toml            own uv project (Python 3.13, mjlab)
├── src/spirob_rl/
│   ├── cli.py train.py       entrypoints: register the tasks, then call
│   │   play.py run.py         mjlab's own train/play; default --log-root to build/
│   ├── infer.py              load a checkpoint and detach it from the sim env
│   ├── tasks/spirob/         THE TASK FAMILY (taken 1:1 from RL_explor)
│   │   ├── base_env_cfg.py     entity, action, sensor ladder, DR, PPO config
│   │   ├── reach_ shape_ trajectory_ wrap_env_cfg.py   one file per goal
│   │   ├── mdp/                commands, rewards, observations, events, curriculum
│   │   └── holdable_poses.npz  measured table of statically holdable poses
│   └── rig/                  hardware bridge + workspace analysis
│       └── acc_board/          joint angles from the 14-accelerometer board
└── tests/
```

The MuJoCo model is **not** in this folder: it is
[`models/spirob_13seg_rl.xml`](../models/README.md), tracked with every other
model in the repository and resolved through `spirob.paths.RL_MODEL`.

## What was changed when the task was imported

The task family comes from the RL_explor repository it was developed in and is
taken **1:1** — same commands, rewards, observations, domain randomisation, PPO
configuration. Deliberately unchanged as well: the task ids
(`RlExplor-Spirob-*`) and the experiment names (`rl_explor_spirob_*`), because
those are the directory names existing checkpoints live under. Renaming one
orphans every run trained before the rename.

What *did* change, and why:

| Change | Why |
|---|---|
| package renamed to `spirob_rl` | a top-level `spirob` package here would shadow the installed sysid library |
| `spirob.xml` → `models/spirob_13seg_rl.xml`, reached via `spirob.paths.RL_MODEL` | models are tracked in `models/` and paths come from `spirob.paths` — a repository non-negotiable |
| training logs → `build/rl/logs`, figures → `build/rl/…` | everything generated lands in `build/`, which is git-ignored |
| figure language defaults to English, `SPIROB_FIG_LOCALE=de` switches | the repository is English; that variable is its one localisation mechanism |
| the ESP32 firmware and the onboard-policy export were not imported | they belong to the rig repository and are not needed to train or to drive the robot from the host |

## The task family

Every variant is registered once per sensor level, e.g.
`RlExplor-Spirob-Tcp-Reach-Imu`. The bare id is the `tendon` level.

| Variant | Task id prefix | Goal |
|---|---|---|
| Reach | `RlExplor-Spirob-Tcp-Reach` | hold the TCP at a static random target on the reachable shell |
| Shape | `RlExplor-Spirob-Shape` | hit a TCP *and* a mid-chain target — command the whole posture |
| Trajectory | `RlExplor-Spirob-Trajectory` | follow a target sweeping along the arc, with preview points |
| Wrap | `RlExplor-Spirob-Wrap` | coil around a randomly placed, randomly sized cylinder |

Sensor ladder (actor only — the critic always gets joint angles, velocities and
TCP position, so a comparison across levels isolates the sensor suite):

| Suffix | Level | Actor sees | On the real rig |
|---|---|---|---|
| `-Force` | force | target + last action only | ✓ motor board alone |
| *(none)* | tendon | + spool encoders (tendon length, velocity) | ✓ motor board alone |
| `-Imu` | imu | + inclination of all 14 segments | ✓ motor + accelerometer board |
| `-Joints` | joints | 13 joint angles and velocities | ✓ motor + accelerometer board |
| `-Oracle` | oracle | + TCP position | ✗ not measurable |

The action is always the same: two tendon forces, `[-1, 1]` mapped onto
`[-150, 0]` N — **pull only**, a positive value is a silent no-op.

## Domain randomisation

`ENABLE_DOMAIN_RANDOMIZATION` / `ENABLE_DR_CURRICULUM` at the top of
[`base_env_cfg.py`](src/spirob_rl/tasks/spirob/base_env_cfg.py) are the two
switches; the final widths live in `DR_TARGETS` in
[`mdp/constants.py`](src/spirob_rl/tasks/spirob/mdp/constants.py). Every term
uses `operation="scale"` against the XML default, so the per-segment spread the
model carries is preserved instead of being flattened by an absolute range, and
the curriculum widens each term from no-op (1.0, 1.0) toward its target over the
first 5000 policy steps.

Flipping `ENABLE_DOMAIN_RANDOMIZATION` to `False` and training the same task
again is the with-DR / without-DR pair; nothing else in the config changes.

## Where the numbers come from

`models/spirob_13seg_rl.xml` is an identified model — a *different* real-to-sim
run than `models/spirob_13seg_identified.xml`, kept as its own file because
swapping it silently changes every trained policy's dynamics. The identification
found that such a fit reproduces gross motion but not the trajectory, and that
the fitted parameters are physically implausible; the gap is
[structural](../docs/sysid/results.md), not an optimisation failure. That is
exactly why DR is on by default here rather than being an afterthought.

## Further reading

* [`docs/rl/`](../docs/rl/index.md) — the long-form write-up.
* [`src/spirob_rl/tasks/spirob/WRAP_TASK.md`](src/spirob_rl/tasks/spirob/WRAP_TASK.md)
  — the wrap task in detail (German).
* [`src/spirob_rl/rig/COMMUNICATION_PROTOCOL.md`](src/spirob_rl/rig/COMMUNICATION_PROTOCOL.md)
  — the serial protocol shared with the ESP32 firmware.
