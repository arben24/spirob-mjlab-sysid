# Training

## TL;DR

```bash
cd rl
uv sync                                          # Python 3.13 venv, ~4 GB
uv run train RlExplor-Spirob-Tcp-Reach --env.scene.num-envs 4096
uv run play  RlExplor-Spirob-Tcp-Reach --checkpoint-file ../build/rl/logs/<exp>/<run>/model_499.pt
```

Checkpoints land in `build/rl/logs/<experiment_name>/<timestamp>/`, alongside
everything else this repository generates. Needs an NVIDIA GPU: MuJoCo-Warp is
the simulator.

## Why `rl/` is its own uv project

mjlab pulls torch, CUDA and MuJoCo-Warp — several GB — and was developed against
Python 3.13. The identification half is pinned to 3.10, and its CI matrix runs
3.10 and 3.12. One environment could not hold both without forcing the whole
sysid setup onto the RL stack's dependencies.

So `rl/` has its own `pyproject.toml`, its own `.python-version` (3.13) and its
own `.venv`. It depends on the root package editable (`path = ".."`), which is
what makes `spirob.paths` available: the tasks resolve the model as
`spirob.paths.RL_MODEL` and every script writes under `build/rl/`, exactly like
every script in `sysid/` writes under `build/`.

```bash
cd rl
uv sync                                # core: mjlab, torch, MuJoCo-Warp
uv sync --extra hardware --extra dev   # + pyserial, pytest, ruff
```

`uv run` from the repository root still uses the 3.10 sysid environment; `uv
run` from `rl/` uses the 3.13 one. Nothing is shared but the source tree.

## Running a training

```bash
uv run train <task-id> [--env.<field> ...] [--agent.<field> ...]
```

The entrypoint registers all 40 SpiRob task ids and then hands over to mjlab's
own `train` script, so every configuration field is reachable from the command
line — `tyro` derives the flags from the config dataclasses.

| Flag | Effect |
|---|---|
| `--env.scene.num-envs 4096` | parallel environments; the main throughput knob |
| `--env.scene.env-spacing 0.5` | grid spacing used by the layout event when watching several envs |
| `--agent.max-iterations 500` | policy iterations (× 128 steps per env) |
| `--agent.logger tensorboard` | mjlab logs to **wandb** by default and will ask you to log in |
| `--log-root <dir>` | overrides the `build/rl/logs` default |

The PPO configuration is shared by every variant (`make_ppo_runner_cfg`):
actor `256-256-256-128-128`, critic `128-128-128-128-64`, ELU, a scalar Gaussian
with `init_std = 2.0`, `learning_rate = 1e-3` on an adaptive schedule with
`desired_kl = 0.01`, `entropy_coef = 0.05`, 500 iterations × 128 steps.

Each sensor level gets its own experiment name — `rl_explor_spirob_tcp_imu`,
`rl_explor_spirob_wrap_joints`, … — so the ablation runs never share a log
directory.

## Watching a policy

```bash
uv run play RlExplor-Spirob-Tcp-Reach --checkpoint-file <path>/model_499.pt
uv run play RlExplor-Spirob-Tcp-Reach-DrPlay --checkpoint-file <path>/model_499.pt
```

Play turns off observation corruption and drops the curriculum. What happens to
the physics then is the difference between the two ids above:

* the plain id evaluates at the **nominal** XML values — every scale-based DR
  event term is removed, so nothing perturbs the dynamics, and the episode is
  effectively endless;
* the `-DrPlay` twin pins every active DR term to its **full-width** target and
  shortens the episode to 20 s with four environments, so each reset draws fresh
  dynamics and several draws are visible side by side.

Same checkpoint, same policy — only what it is being asked to cope with differs.

## The with-DR / without-DR pair

The experiment the identification result calls for is a pair of runs on one
task: one with randomisation, one without. Only the switches at the top of
`base_env_cfg.py` change between them:

```python
ENABLE_DOMAIN_RANDOMIZATION = False   # nominal dynamics in every environment
```

With it `False`, both the DR event terms and the curriculum are dropped
entirely. With it `True` and `ENABLE_DR_CURRICULUM = False`, every term sits at
its full `DR_TARGETS` width from step 0 instead of being phased in.

## Regenerating the pose table

The shape task samples its targets from `holdable_poses.npz`, a measured grid
over the two tendon commands. Regenerate it after changing the model or its
dynamic parameters:

```bash
uv run python -m spirob_rl.tasks.spirob._make_pose_table
```

It runs 24 × 24 environments for 8 s each, averages the pose over the last
second and reports the residual oscillation — the base joint is almost undamped
in the model (`damping = 1e-4`), so some poses keep ringing at ~0.011 rad rather
than coming fully to rest, and the recorded target is the pose the tentacle
oscillates *about*.

## Tests

```bash
cd rl && uv run --extra dev pytest
```

These construct real environments on the GPU, so they are not part of the
repository's CI (which has no GPU and does not install mjlab). What CI *does*
check is the model contract the tasks depend on — tendon names, `site_tcp`, the
14 `site_imu_*` sites, the pull-only `ctrlrange`, the rest-pose tendon length —
in `tests/test_model.py`, using MuJoCo alone.
