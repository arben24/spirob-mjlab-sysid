"""Spirob reach task: hold the TCP at a static random target.

A random TCP (tip of the last segment ``seg_0``) target position is sampled on
the reachable workspace shell and the policy has to bend the tentacle so the
TCP arrives there and stays.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, joint_vel_l2
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from . import mdp
from .base_env_cfg import (
  SensorLevel,
  apply_play_overrides,
  make_base_env_cfg,
  make_ppo_runner_cfg,
)
from .mdp.constants import JOINTS_CFG

COMMAND_NAME = "tcp_target"


def make_env_cfg(
  sensor_level: SensorLevel = "tendon",
  history_length: int = 10,
) -> ManagerBasedRlEnvCfg:
  commands = {
    COMMAND_NAME: mdp.TcpPositionCommandCfg(
      resampling_time_range=(12.0, 15.0),
      debug_vis=True,
    ),
  }

  rewards = {
    "tcp_tracking": RewardTermCfg(
      func=mdp.position_tracking,
      weight=1.0,
      params={"std": 0.2, "command_name": COMMAND_NAME},
    ),
    "tcp_tracking_fine": RewardTermCfg(
      func=mdp.position_tracking,
      weight=3.0,
      params={"std": 0.05, "command_name": COMMAND_NAME},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    # Keeps the policy mean inside the saturating action range. See action_l2.
    "action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.02),
    "joint_vel": RewardTermCfg(
      func=joint_vel_l2,
      weight=-0.005,
      params={"asset_cfg": JOINTS_CFG},
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


def reach_env_cfg(
  play: bool = False,
  sensor_level: SensorLevel = "tendon",
  history_length: int = 5,
  dr_in_play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_env_cfg(sensor_level=sensor_level, history_length=history_length)
  if play:
    apply_play_overrides(cfg, dr_in_play=dr_in_play)
  return cfg


def reach_ppo_runner_cfg(
  sensor_level: SensorLevel = "tendon",
) -> RslRlOnPolicyRunnerCfg:
  # One directory per sensor level so the ablation runs stay separated. The name
  # is unchanged from before the task-family split, so existing checkpoints in
  # logs/rsl_rl/rl_explor_spirob_tcp_<level>/ keep resolving.
  return make_ppo_runner_cfg(f"rl_explor_spirob_tcp_{sensor_level}")
