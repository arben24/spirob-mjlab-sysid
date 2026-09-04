"""Spirob shape task: command the whole posture, not just the endpoint.

Two targets are given at once -- one for the TCP and one for a mid-chain
segment -- so the policy has to reproduce a commanded *shape* rather than
merely getting the tip to a point. Because a tendon-driven, inextensible chain
cannot reach arbitrary point pairs, both targets are read off one sampled joint
configuration via forward kinematics (see :class:`~.mdp.commands.ShapeCommand`).

The mid target is deliberately rewarded more loosely than the tip: the tip is
the thing the task is ultimately about, and an over-tight mid term would fight
it for the same two tendons.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from . import mdp
from .base_env_cfg import (
  SensorLevel,
  apply_play_overrides,
  make_base_env_cfg,
  make_ppo_runner_cfg,
)
from .mdp.constants import MID_SITE_CFG

COMMAND_NAME = "shape_target"

# Index into the command's tracked points, matching ShapeCommand.site_names.
TIP_POINT = 0
MID_POINT = 1


def make_env_cfg(
  sensor_level: SensorLevel = "tendon",
  history_length: int = 10,
) -> ManagerBasedRlEnvCfg:
  commands = {
    COMMAND_NAME: mdp.ShapeCommandCfg(
      resampling_time_range=(12.0, 15.0),
      debug_vis=True,
    ),
  }

  rewards = {
    # Coarse term over both points: gets the overall posture into the right
    # region before either point is precise.
    "shape_tracking": RewardTermCfg(
      func=mdp.position_tracking,
      weight=1.0,
      params={"std": 0.2, "command_name": COMMAND_NAME},
    ),
    # Tip: the primary goal, tightest kernel and highest weight.
    "tip_tracking_fine": RewardTermCfg(
      func=mdp.position_tracking,
      weight=3.0,
      params={"std": 0.05, "command_name": COMMAND_NAME, "point_idx": TIP_POINT},
    ),
    # Mid segment: shapes the body, but with a looser kernel and lower weight so
    # it guides rather than competes with the tip.
    "mid_tracking_fine": RewardTermCfg(
      func=mdp.position_tracking,
      weight=1.5,
      params={"std": 0.08, "command_name": COMMAND_NAME, "point_idx": MID_POINT},
    ),
  }

  # The critic also gets the measured mid-segment position, since that is part
  # of what it has to value here.
  extra_critic_terms = {
    "mid_pos": ObservationTermCfg(
      func=mdp.site_pos_rel,
      params={"asset_cfg": MID_SITE_CFG},
    ),
  }

  return make_base_env_cfg(
    commands=commands,
    rewards=rewards,
    command_name=COMMAND_NAME,
    sensor_level=sensor_level,
    history_length=history_length,
    episode_length_s=50.0,
    extra_critic_terms=extra_critic_terms,
  )


def shape_env_cfg(
  play: bool = False,
  sensor_level: SensorLevel = "tendon",
  history_length: int = 5,
  dr_in_play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_env_cfg(sensor_level=sensor_level, history_length=history_length)
  if play:
    apply_play_overrides(cfg, dr_in_play=dr_in_play)
  return cfg


def shape_ppo_runner_cfg(
  sensor_level: SensorLevel = "tendon",
) -> RslRlOnPolicyRunnerCfg:
  return make_ppo_runner_cfg(f"rl_explor_spirob_shape_{sensor_level}")
