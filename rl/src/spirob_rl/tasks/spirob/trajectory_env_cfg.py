"""Spirob trajectory task: follow a continuously moving TCP target.

Instead of jumping to a new static point every few seconds, the target sweeps
along the reachable arc, so the policy has to *track* a moving goal rather than
settle on a fixed one. The command also reports a few preview points, which is
what allows a policy to lead the target instead of lagging behind it.

Shorter resampling than the reach task: one resample draws a whole new sweep
(center, amplitude, frequency), and several sweeps per episode give more variety
than a single long one.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from . import mdp
from .base_env_cfg import (
  SensorLevel,
  apply_play_overrides,
  make_base_env_cfg,
  make_ppo_runner_cfg,
)

COMMAND_NAME = "traj_target"


def make_env_cfg(
  sensor_level: SensorLevel = "tendon",
  history_length: int = 10,
  num_preview: int = 3,
) -> ManagerBasedRlEnvCfg:
  commands = {
    COMMAND_NAME: mdp.TrajectoryCommandCfg(
      resampling_time_range=(10.0, 15.0),
      debug_vis=True,
      num_preview=num_preview,
    ),
  }

  rewards = {
    "traj_tracking": RewardTermCfg(
      func=mdp.position_tracking,
      weight=1.0,
      params={"std": 0.2, "command_name": COMMAND_NAME},
    ),
    "traj_tracking_fine": RewardTermCfg(
      func=mdp.position_tracking,
      weight=3.0,
      params={"std": 0.05, "command_name": COMMAND_NAME},
    ),
  }

  return make_base_env_cfg(
    commands=commands,
    rewards=rewards,
    command_name=COMMAND_NAME,
    sensor_level=sensor_level,
    history_length=history_length,
    episode_length_s=50.0,
  )


def trajectory_env_cfg(
  play: bool = False,
  sensor_level: SensorLevel = "tendon",
  history_length: int = 5,
  dr_in_play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_env_cfg(sensor_level=sensor_level, history_length=history_length)
  if play:
    apply_play_overrides(cfg, dr_in_play=dr_in_play)
  return cfg


def trajectory_ppo_runner_cfg(
  sensor_level: SensorLevel = "tendon",
) -> RslRlOnPolicyRunnerCfg:
  return make_ppo_runner_cfg(f"rl_explor_spirob_traj_{sensor_level}")
