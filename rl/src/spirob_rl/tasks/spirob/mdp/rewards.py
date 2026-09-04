"""Reward terms for the spirob task family."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .constants import IMU_CFG

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor


def position_tracking(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  point_idx: int | None = None,
) -> torch.Tensor:
  """Gaussian kernel over the tracked-point-to-target distance.

  Works with any :class:`PointTrackingCommand`, which exposes targets and the
  matching measured points as ``[num_envs, num_points, 3]``. With
  ``point_idx=None`` the squared errors of all tracked points are averaged
  before the kernel is applied, so a single-point task (reach, trajectory)
  reduces exactly to the plain TCP kernel. Pass ``point_idx`` to weight one
  point separately -- e.g. a tighter ``std`` on the TCP than on the mid
  segment in the shape task.
  """
  command = env.command_manager.get_term(command_name)
  error = torch.sum(
    torch.square(command.target_pos_w - command.tracked_pos_w), dim=-1
  )
  error = error[:, point_idx] if point_idx is not None else error.mean(dim=-1)
  return torch.exp(-error / std**2)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared magnitude of the raw policy action.

  The action term saturates outside [-1, 1] (it maps to the tendon ctrlrange),
  so beyond that the physics no longer changes and nothing pulls the Gaussian
  mean back. Without this penalty the mean random-walks outward and the
  action-rate term blows up mid-training.
  """
  return torch.sum(torch.square(env.action_manager.action), dim=-1)


def _segment_offsets_from_object(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Segment site positions relative to the commanded object center.

  Returns ``(offsets, radius)`` where ``offsets`` is ``[num_envs, num_sites, 3]``
  (env-origin-relative, so it cancels the same way everywhere else in this
  file) and ``radius`` is ``[num_envs]``. Shared by both wrap reward terms so
  they always agree on which points and which object they're scoring against.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  sites = asset.data.site_pos_w[:, asset_cfg.site_ids] - env.scene.env_origins.unsqueeze(1)
  center = command.target_pos_w - env.scene.env_origins
  return sites - center.unsqueeze(1), command.target_radius


def wrap_proximity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = IMU_CFG,
) -> torch.Tensor:
  """Mean Gaussian kernel of each segment site's distance to the object's
  surface, averaged over all tracked sites.

  Unlike a single-point reach reward, averaging over every segment (not just
  the tip) rewards the whole body hugging the surface -- a prerequisite for
  wrapping, not just touching the object with one point.
  """
  offsets, radius = _segment_offsets_from_object(env, command_name, asset_cfg)
  surface_dist = torch.norm(offsets, dim=-1) - radius.unsqueeze(-1)
  return torch.exp(-torch.square(surface_dist) / std**2).mean(dim=-1)


def wrap_coverage(
  env: ManagerBasedRlEnv,
  proximity_std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = IMU_CFG,
) -> torch.Tensor:
  """Reward the segment sites near the object's surface for spreading around
  it, not clustering on one side.

  Weights every site by the same proximity kernel as :func:`wrap_proximity`,
  then computes the resultant length of the weighted mean unit vector from the
  object center to each site (circular statistics): full angular spread makes
  those vectors cancel out (resultant length -> 0), all sites on one side make
  them reinforce (resultant length -> 1). ``1 - resultant_length`` is 0 for no
  coverage and grows monotonically with how far the wrap has curled around the
  object -- an actual encirclement, not merely a proximity check repeated N
  times. Scaled by the mean proximity weight so it stays near 0 when nothing is
  close to the object at all (otherwise sites scattered far away in an
  accidentally even spread would score well for the wrong reason).
  """
  offsets, radius = _segment_offsets_from_object(env, command_name, asset_cfg)
  dist = torch.norm(offsets, dim=-1)
  weight = torch.exp(-torch.square(dist - radius.unsqueeze(-1)) / proximity_std**2)
  weight_sum = weight.sum(dim=-1).clamp_min(1e-6)

  theta = torch.atan2(offsets[..., 0], offsets[..., 2])
  resultant_x = (weight * torch.cos(theta)).sum(dim=-1) / weight_sum
  resultant_z = (weight * torch.sin(theta)).sum(dim=-1) / weight_sum
  resultant_len = torch.sqrt(resultant_x**2 + resultant_z**2)

  coverage = 1.0 - resultant_len
  return coverage * (weight_sum / offsets.shape[1])


def wrap_force_distribution(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_total_force: float = 1e-3,
) -> torch.Tensor:
  """Reward the *contact force*, not just position, being spread across many
  segments instead of concentrated on one or two.

  ``wrap_coverage`` already rewards segments being geometrically spread around
  the object, but says nothing about how hard each one is actually pressing --
  a policy could satisfy it with thirteen segments barely grazing the surface
  and one segment doing all the gripping. This term reads real per-segment
  contact forces from ``sensor_name`` (a :class:`ContactSensor` between the
  spirob's segment geoms and the object, see ``wrap_env_cfg.py``) and scores
  the *shape* of that force distribution directly.

  Treats each segment's contact-force magnitude as an unnormalized
  distribution and returns its Shannon entropy, normalized by ``log(P)`` (P =
  number of segments the sensor tracks) so the result is in ``[0, 1]``:

  * all force on one segment -> entropy 0 (minimum, regardless of P).
  * force split evenly across ``k`` segments -> entropy ``log(k) / log(P)``,
    which increases both as force evens out across the *current* contacts and
    as *more* segments join them -- one formula rewards both "spread it out"
    and "grip with more of the body" without needing two separate terms.

  Scale-invariant (built from normalized proportions), so this doesn't reward
  gripping harder overall, only distributing whatever force is already there
  more evenly. Returns 0 (not NaN) when total contact force is ~0.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None, f"Sensor '{sensor_name}' must request the 'force' field."
  magnitude = torch.norm(force, dim=-1)  # [B, P]

  total = magnitude.sum(dim=-1)
  has_contact = total > min_total_force
  proportion = magnitude / total.clamp_min(1e-8).unsqueeze(-1)

  # lim_{x->0} x * log(x) = 0, so zero-force segments drop out cleanly.
  log_proportion = torch.where(
    proportion > 0, torch.log(proportion.clamp_min(1e-12)), torch.zeros_like(proportion)
  )
  entropy = -(proportion * log_proportion).sum(dim=-1)

  max_entropy = math.log(magnitude.shape[-1])
  normalized_entropy = entropy / max_entropy
  return torch.where(has_contact, normalized_entropy, torch.zeros_like(normalized_entropy))
