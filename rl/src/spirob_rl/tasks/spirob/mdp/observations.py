"""Observation terms for the spirob task family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .constants import IMU_CFG, TCP_CFG, TENDON_REST_LEN, TENDONS_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def site_pos_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Site positions relative to the env origin. Shape: [num_envs, 3 * num_sites].

  The env origin offset cancels out of every difference we compute (target minus
  measured), so the grid layout used for visualization never changes the numbers
  the policy sees.
  """
  asset: Entity = env.scene[asset_cfg.name]
  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  return (pos_w - env.scene.env_origins.unsqueeze(1)).flatten(1)


def tcp_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = TCP_CFG,
) -> torch.Tensor:
  """TCP position relative to the env origin. Shape: [num_envs, 3]."""
  return site_pos_rel(env, asset_cfg)


def tendon_len_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = TENDONS_CFG,
) -> torch.Tensor:
  """Tendon lengths relative to the rest configuration. Shape: [num_envs, 2].

  On the real rig this is what the spool encoders measure.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_len[:, asset_cfg.tendon_ids] - TENDON_REST_LEN


def tendon_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = TENDONS_CFG,
) -> torch.Tensor:
  """Tendon velocities. Shape: [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_vel[:, asset_cfg.tendon_ids]


def segment_pitch_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = IMU_CFG,
) -> torch.Tensor:
  """Cos/sin of each segment's pitch angle. Shape: [num_envs, 2 * num_sites].

  Stand-in for the per-segment IMUs declared in the XML: motion is planar
  around the y axis, so a fused IMU reading reduces to the segment's
  inclination. Taken from the site orientation quaternion (w, x, y, z).
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.site_quat_w[:, asset_cfg.site_ids]
  w, y = quat[..., 0], quat[..., 2]
  # Planar rotation about y: pitch = 2 * atan2(y, w).
  pitch = 2.0 * torch.atan2(y, w)
  return torch.cat([torch.cos(pitch), torch.sin(pitch)], dim=-1)
