"""Guard-rails for the SpiRob task family.

These need a GPU (MuJoCo-Warp is the simulator) and the `rl/` environment, so
they are not part of the repository's CI. What CI checks instead is the model
contract the tasks depend on -- see `tests/test_model.py` at the repository
root, which needs nothing but MuJoCo.

Run with: `cd rl && uv run --extra dev pytest`
"""

from __future__ import annotations

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from spirob.paths import RL_MODEL

import spirob_rl  # noqa: F401  (registers the tasks)
from spirob_rl.tasks.spirob.mdp.constants import TENDON_CTRL_RANGE

VARIANTS = ("Tcp-Reach", "Shape", "Trajectory", "Wrap")
LEVEL_SUFFIXES = ("-Force", "", "-Imu", "-Joints", "-Oracle")

requires_cuda = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="mjlab needs a CUDA device"
)


def test_every_variant_is_registered_at_every_sensor_level():
  tasks = set(list_tasks())
  for variant in VARIANTS:
    for suffix in LEVEL_SUFFIXES:
      task_id = f"RlExplor-Spirob-{variant}{suffix}"
      assert task_id in tasks
      # Each one also has a DR-play twin, sharing its checkpoints.
      assert f"{task_id}-DrPlay" in tasks


def test_the_tasks_train_on_the_tracked_model():
  """The model lives in `models/`, resolved through `spirob.paths` -- not next
  to the task, and not hard-coded."""
  assert RL_MODEL.exists()
  from spirob_rl.tasks.spirob.mdp.constants import SPIROB_XML

  assert SPIROB_XML == RL_MODEL


def test_experiment_names_stay_stable():
  """Checkpoints live under `build/rl/logs/<experiment_name>/`, so renaming one
  orphans every run trained before the rename."""
  assert load_rl_cfg("RlExplor-Spirob-Tcp-Reach-Imu").experiment_name == (
    "rl_explor_spirob_tcp_imu"
  )
  assert load_rl_cfg("RlExplor-Spirob-Wrap").experiment_name == (
    "rl_explor_spirob_wrap_tendon"
  )


def test_the_action_stays_pull_only():
  cfg = load_env_cfg("RlExplor-Spirob-Tcp-Reach")
  action = cfg.actions["tendon_force"]
  lo, hi = TENDON_CTRL_RANGE
  assert lo < 0 and hi == 0
  # ctrl = scale * a + offset, so the policy's [-1, 1] maps exactly onto it.
  assert action.offset - action.scale == pytest.approx(lo)
  assert action.offset + action.scale == pytest.approx(hi)


def test_play_drops_the_curriculum_and_the_randomization():
  play = load_env_cfg("RlExplor-Spirob-Tcp-Reach", play=True)
  assert play.curriculum == {}
  assert not play.observations["actor"].enable_corruption
  assert all("operation" not in term.params for term in play.events.values())


def test_dr_play_keeps_the_randomization_at_full_width():
  from spirob_rl.tasks.spirob.mdp.constants import DR_TARGETS

  play = load_env_cfg("RlExplor-Spirob-Tcp-Reach-DrPlay", play=True)
  randomized = {n: t for n, t in play.events.items() if "operation" in t.params}
  assert randomized, "DR-play must keep the randomization events"
  for name, term in randomized.items():
    assert term.params["ranges"] == DR_TARGETS[name]


@requires_cuda
@pytest.mark.parametrize("task_id", ["RlExplor-Spirob-Tcp-Reach-Imu", "RlExplor-Spirob-Wrap"])
def test_env_builds_and_steps(task_id):
  """Construction is where a broken sensor term, entity or contact sensor shows
  up -- and the wrap task additionally exercises the object entity and the
  contact sensor."""
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    for _ in range(5):
      obs, reward, _, _, _ = env.step(torch.zeros(2, 2, device="cuda:0"))
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    # The actor's width is the sensor level; the critic's never changes.
    assert obs["actor"].shape[0] == 2
  finally:
    env.close()
