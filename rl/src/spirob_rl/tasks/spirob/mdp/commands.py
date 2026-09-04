"""Command terms for the spirob task family.

All three variants track points in the x-z bending plane, so they share a small
base class: a command owns ``target_pos_w`` and the matching measured
``tracked_pos_w``, both ``[num_envs, num_points, 3]``, and the generic
:func:`~..rewards.position_tracking` reward works off either.

* :class:`TcpPositionCommand` -- one static target for the tip (reach).
* :class:`ShapeCommand` -- targets for the tip *and* a mid-chain segment, drawn
  from one forward-kinematics pose so the pair is always achievable.
* :class:`TrajectoryCommand` -- one target that sweeps continuously along the
  reachable arc, optionally with preview points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

from .constants import (
  ENTITY_NAME,
  HOLDABLE_POSES_NPZ,
  MID_SEGMENT,
  SPIROB_XML,
)
from .kinematics import HoldablePoseTable, PlanarChain
from .object_spec import OBJECT_HALF_LENGTH

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Colors used for the debug spheres, by tracked-point index.
_POINT_COLORS = ((1.0, 0.5, 0.0, 0.5), (0.2, 0.6, 1.0, 0.5))


def shell_radius_at(
  angle: torch.Tensor, shell_radius: float, shell_curvature: float
) -> torch.Tensor:
  """Radius of the reachable workspace shell at a given angle from +z.

  The tentacle is inextensible, so its quasi-static workspace is a thin
  arc-shaped shell: bending sideways moves the tip to larger angles and smaller
  radii. Fitted to a quasi-static random-actuation probe.
  """
  return shell_radius - shell_curvature * angle**2


def polar_to_xz(radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Convert shell polar coordinates to a position in the x-z plane. ``[N, 3]``."""
  return torch.stack(
    [radius * torch.sin(angle), torch.zeros_like(angle), radius * torch.cos(angle)],
    dim=-1,
  )


class PointTrackingCommand(CommandTerm):
  """Base for commands that ask the spirob to put N sites at N target points."""

  cfg: PointTrackingCommandCfg

  def __init__(self, cfg: PointTrackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    site_ids, _ = self.robot.find_sites(self.site_names, preserve_order=True)
    self._site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    self.num_points = len(site_ids)

    self.target_pos_w = torch.zeros(
      self.num_envs, self.num_points, 3, device=self.device
    )
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def site_names(self) -> tuple[str, ...]:
    """Sites tracked by this command, in target order."""
    raise NotImplementedError

  @property
  def tracked_pos_w(self) -> torch.Tensor:
    """Measured positions of the tracked sites. ``[num_envs, num_points, 3]``."""
    return self.robot.data.site_pos_w[:, self._site_ids]

  @property
  def command(self) -> torch.Tensor:
    """Targets relative to the env origin, flattened. ``[num_envs, 3 * num_points]``."""
    return (self.target_pos_w - self._env.scene.env_origins.unsqueeze(1)).flatten(1)

  def _update_metrics(self) -> None:
    distances = torch.norm(self.target_pos_w - self.tracked_pos_w, dim=-1)
    # Worst tracked point decides success, so a good tip cannot mask a bad body.
    self.metrics["position_error"] = distances.mean(dim=-1)
    self.metrics["at_goal"] = (
      distances.max(dim=-1).values < self.cfg.success_threshold
    ).float()

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      for point in range(self.num_points):
        visualizer.add_sphere(
          center=self.target_pos_w[batch, point].cpu().numpy(),
          radius=0.01,
          color=_POINT_COLORS[point % len(_POINT_COLORS)],
          label=f"target_{point}_{batch}",
        )


@dataclass(kw_only=True)
class PointTrackingCommandCfg(CommandTermCfg):
  entity_name: str = ENTITY_NAME
  success_threshold: float = 0.02


##
# Reach: a single static TCP target.
##


class TcpPositionCommand(PointTrackingCommand):
  """Random static TCP target positions on the reachable workspace shell."""

  cfg: TcpPositionCommandCfg

  @property
  def site_names(self) -> tuple[str, ...]:
    return (self.cfg.site_name,)

  @property
  def tcp_pos_w(self) -> torch.Tensor:
    return self.tracked_pos_w[:, 0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    angle = sample_uniform(
      self.cfg.angle_range[0], self.cfg.angle_range[1], (n,), device=self.device
    )
    radius = shell_radius_at(
      angle, self.cfg.shell_radius, self.cfg.shell_curvature
    ) + sample_uniform(
      -self.cfg.shell_band, self.cfg.shell_band, (n,), device=self.device
    )
    target = polar_to_xz(radius, angle)
    self.target_pos_w[env_ids, 0] = target + self._env.scene.env_origins[env_ids]


@dataclass(kw_only=True)
class TcpPositionCommandCfg(PointTrackingCommandCfg):
  site_name: str = "site_tcp"
  # Workspace shell in the x-z plane, relative to the base. Values fitted to a
  # quasi-static random-actuation probe: holdable TCP positions lie within
  # +/- shell_band of r(angle) = shell_radius - shell_curvature * angle**2
  # for angles (from the +z axis) up to ~1.7 rad in both directions.
  angle_range: tuple[float, float] = (-1.7, 1.7)
  shell_radius: float = 0.33
  shell_curvature: float = 0.045
  shell_band: float = 0.08

  def build(self, env: ManagerBasedRlEnv) -> TcpPositionCommand:
    return TcpPositionCommand(self, env)


##
# Shape: TCP plus a mid-chain segment, from one consistent pose.
##


class ShapeCommand(PointTrackingCommand):
  """Targets for the tip *and* a mid-chain segment, so the whole posture is
  commanded rather than just the endpoint.

  Two independently drawn points would nearly always be unreachable together
  (the chain is inextensible and only two tendons actuate it). Instead a joint
  configuration is drawn from the smooth curvature family the tendons can
  actually hold, and both targets are read off its forward kinematics -- so
  every commanded pair lies on one achievable pose by construction.
  """

  cfg: ShapeCommandCfg

  def __init__(self, cfg: ShapeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.chain = PlanarChain(SPIROB_XML, self.device)
    self.poses = HoldablePoseTable(HOLDABLE_POSES_NPZ, self.device)

  @property
  def site_names(self) -> tuple[str, ...]:
    return (self.cfg.tip_site_name, f"site_imu_{self.cfg.mid_segment}")

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    q = self.poses.sample(n, jitter=self.cfg.jitter)
    origins, thetas = self.chain.forward(q)
    tip = self.chain.tcp_pos(origins, thetas)
    mid = self.chain.site_pos(origins, thetas, self.cfg.mid_segment)

    env_origins = self._env.scene.env_origins[env_ids]
    self.target_pos_w[env_ids, 0] = tip + env_origins
    self.target_pos_w[env_ids, 1] = mid + env_origins


@dataclass(kw_only=True)
class ShapeCommandCfg(PointTrackingCommandCfg):
  tip_site_name: str = "site_tcp"
  mid_segment: int = MID_SEGMENT
  # Per-joint noise added on top of a sampled holdable pose, in rad. Kept at the
  # scale of the poses' own residual oscillation (~0.011 rad) so targets stay
  # achievable; raise it to demand generalization beyond the measured manifold.
  jitter: float = 0.01
  # Two points must both be hit, so allow a little more slack than the tip-only
  # reach task.
  success_threshold: float = 0.03

  def build(self, env: ManagerBasedRlEnv) -> ShapeCommand:
    return ShapeCommand(self, env)


##
# Trajectory: a continuously moving TCP target.
##


class TrajectoryCommand(PointTrackingCommand):
  """A TCP target that sweeps continuously along the reachable arc.

  Instead of jumping to a new static point every few seconds, the target angle
  follows a sinusoid and the radius rides the workspace shell, so the commanded
  point stays reachable at every instant and the policy has to *follow* rather
  than settle. A new sweep (center, amplitude, frequency, phase) is drawn on
  each resample.

  With ``num_preview > 0`` the command also reports where the target will be a
  few steps ahead, which is what lets a policy lead a moving goal instead of
  chasing it.
  """

  cfg: TrajectoryCommandCfg

  def __init__(self, cfg: TrajectoryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._phase = torch.zeros(self.num_envs, device=self.device)
    self._center = torch.zeros(self.num_envs, device=self.device)
    self._amplitude = torch.zeros(self.num_envs, device=self.device)
    self._freq = torch.zeros(self.num_envs, device=self.device)
    self._time = torch.zeros(self.num_envs, device=self.device)
    self._preview_pos_w = torch.zeros(
      self.num_envs, max(cfg.num_preview, 1), 3, device=self.device
    )
    self.metrics["target_speed"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def site_names(self) -> tuple[str, ...]:
    return (self.cfg.site_name,)

  @property
  def tcp_pos_w(self) -> torch.Tensor:
    return self.tracked_pos_w[:, 0]

  @property
  def command(self) -> torch.Tensor:
    """Current target plus preview points, relative to the env origin."""
    origin = self._env.scene.env_origins.unsqueeze(1)
    current = (self.target_pos_w - origin).flatten(1)
    if self.cfg.num_preview <= 0:
      return current
    preview = (self._preview_pos_w - origin).flatten(1)
    return torch.cat([current, preview], dim=-1)

  def _target_at(self, time: torch.Tensor) -> torch.Tensor:
    angle = self._center + self._amplitude * torch.sin(
      2.0 * math.pi * self._freq * time + self._phase
    )
    angle = angle.clamp(self.cfg.angle_range[0], self.cfg.angle_range[1])
    radius = shell_radius_at(angle, self.cfg.shell_radius, self.cfg.shell_curvature)
    return polar_to_xz(radius, angle)

  def _write_targets(self, env_ids: torch.Tensor | slice) -> None:
    origins = self._env.scene.env_origins
    self.target_pos_w[:, 0] = self._target_at(self._time) + origins
    for k in range(self.cfg.num_preview):
      ahead = self._time + (k + 1) * self.cfg.preview_dt
      self._preview_pos_w[:, k] = self._target_at(ahead) + origins
    del env_ids  # Sweeps are cheap; recompute for all envs.

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    lo, hi = self.cfg.angle_range
    self._amplitude[env_ids] = sample_uniform(
      self.cfg.amplitude_range[0],
      self.cfg.amplitude_range[1],
      (n,),
      device=self.device,
    )
    # Keep the whole sweep inside the reachable angle band.
    max_center = (hi - self._amplitude[env_ids]).clamp_min(0.0)
    self._center[env_ids] = (
      sample_uniform(-1.0, 1.0, (n,), device=self.device) * max_center
    )
    self._freq[env_ids] = sample_uniform(
      self.cfg.frequency_range[0],
      self.cfg.frequency_range[1],
      (n,),
      device=self.device,
    )
    self._phase[env_ids] = sample_uniform(
      0.0, 2.0 * math.pi, (n,), device=self.device
    )
    self._time[env_ids] = 0.0
    self._write_targets(env_ids)

  def _update_command(self) -> None:
    previous = self.target_pos_w[:, 0].clone()
    self._time += self._env.step_dt
    self._write_targets(slice(None))
    self.metrics["target_speed"] = (
      torch.norm(self.target_pos_w[:, 0] - previous, dim=-1) / self._env.step_dt
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)
    env_indices = visualizer.get_env_indices(self.num_envs)
    for batch in env_indices:
      for k in range(self.cfg.num_preview):
        visualizer.add_sphere(
          center=self._preview_pos_w[batch, k].cpu().numpy(),
          radius=0.005,
          color=(1.0, 1.0, 1.0, 0.25),
          label=f"preview_{k}_{batch}",
        )


@dataclass(kw_only=True)
class TrajectoryCommandCfg(PointTrackingCommandCfg):
  site_name: str = "site_tcp"
  # Same workspace shell as the reach task, so every point of the sweep is
  # reachable at every instant.
  angle_range: tuple[float, float] = (-1.7, 1.7)
  shell_radius: float = 0.33
  shell_curvature: float = 0.045
  # Sweep shape. amplitude is in rad of arc angle, frequency in Hz.
  amplitude_range: tuple[float, float] = (0.4, 1.5)
  frequency_range: tuple[float, float] = (0.05, 0.25)
  # Preview: how many future target points to report, and how far apart.
  num_preview: int = 3
  preview_dt: float = 0.2
  # A moving target is never exactly hit, so success is a looser band. At the
  # 3 cm used for the static tasks this metric reads a flat zero and tells you
  # nothing; 5 cm is roughly "following closely" at the sweep speeds above.
  success_threshold: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> TrajectoryCommand:
    return TrajectoryCommand(self, env)


##
# Wrap: a graspable cylinder to coil around.
##


class WrapCommand(CommandTerm):
  """A randomly placed, randomly sized cylinder for the spirob to wrap around.

  Unlike the other commands, the "target" here is not a site on the spirob --
  it is a real object entity (see ``mdp/object_spec.py``), placed and sized per
  environment on every resample:

  * position, via ``Entity.write_mocap_pose_to_sim`` (the object has no joint,
    so mjlab auto-wraps it in a mocap body -- see ``constants.OBJECT_CFG``);
  * radius, by writing the compiled model's ``geom_size`` field directly for
    that environment, the same per-world-field-expansion technique
    ``mdp.events.grid_layout`` uses for the spirob's own base body.

  ``target_pos_w`` and ``target_radius`` are then the single source of truth
  the wrap-quality rewards compare the spirob's segments against.
  """

  cfg: WrapCommandCfg

  def __init__(self, cfg: WrapCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.object_entity_name]
    geom_ids_local, _ = self.object.find_geoms((cfg.object_geom_name,))
    # find_geoms returns an entity-local index, but env.sim.model.geom_size
    # (and geom_rbound/geom_aabb below) are indexed globally across every
    # entity's geoms in the compiled scene. Writing the local index directly
    # silently targeted global geom 0 -- the spirob's own base segment,
    # seg_13's box geom "g_13" -- instead of the object's cylinder.
    self._geom_id = int(self.object.indexing.geom_ids[geom_ids_local[0]].item())

    self.robot: Entity = env.scene[ENTITY_NAME]
    imu_site_ids, _ = self.robot.find_sites(
      tuple(f"site_imu_{i}" for i in range(14))
    )
    self._imu_site_ids = torch.tensor(
      imu_site_ids, device=self.device, dtype=torch.long
    )

    # geom_rbound (broadphase bounding-sphere radius) and geom_aabb (local
    # AABB half-size) are copied from the compiled MjModel at load time and
    # never recomputed by MuJoCo Warp when geom_size is written directly --
    # mjlab's own dr.geom_size recomputes both after every write for exactly
    # this reason (see mjlab.envs.mdp.dr.geom._recompute_geom_bounds). Left
    # stale, the broadphase keeps pruning contact candidates against the
    # compile-time default radius (0.03 m) instead of the actual sampled one
    # (up to 0.10 m), silently dropping contacts for the larger objects.
    env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))

    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_radius = torch.zeros(self.num_envs, device=self.device)
    self.metrics["min_surface_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Object center (relative to env origin) and radius. ``[num_envs, 4]``."""
    center = self.target_pos_w - self._env.scene.env_origins
    return torch.cat([center, self.target_radius.unsqueeze(-1)], dim=-1)

  def _update_metrics(self) -> None:
    # How close does the nearest segment site get to the object's surface --
    # a simple proxy for "is anything touching it". Wrap *quality* (coverage
    # around the object, not just proximity) is what the reward measures.
    sites = self.robot.data.site_pos_w[:, self._imu_site_ids]
    dist_to_center = torch.norm(sites - self.target_pos_w.unsqueeze(1), dim=-1)
    surface_dist = (dist_to_center - self.target_radius.unsqueeze(-1)).abs()
    min_surface_dist = surface_dist.min(dim=-1).values
    self.metrics["min_surface_dist"] = min_surface_dist
    self.metrics["at_goal"] = (min_surface_dist < self.cfg.success_threshold).float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    angle = sample_uniform(
      self.cfg.angle_range[0], self.cfg.angle_range[1], (n,), device=self.device
    )
    radius = sample_uniform(
      self.cfg.center_radius_range[0],
      self.cfg.center_radius_range[1],
      (n,),
      device=self.device,
    )
    center = polar_to_xz(radius, angle)
    self.target_pos_w[env_ids] = center + self._env.scene.env_origins[env_ids]

    self.target_radius[env_ids] = sample_uniform(
      self.cfg.object_radius_range[0],
      self.cfg.object_radius_range[1],
      (n,),
      device=self.device,
    )

    pose = torch.zeros(n, 7, device=self.device)
    pose[:, 0:3] = self.target_pos_w[env_ids]
    pose[:, 3] = 1.0  # Identity quat: cylinder axis stays along local y.
    self.object.write_mocap_pose_to_sim(pose, env_ids)

    new_radius = self.target_radius[env_ids]
    self._env.sim.model.geom_size[env_ids, self._geom_id, 0] = new_radius

    # Recompute the broadphase bounds MuJoCo Warp doesn't derive on its own
    # (cylinder formulas, see mjlab.envs.mdp.dr.geom._recompute_geom_bounds).
    # The half-length (geom_size[..., 1]) never changes, only the radius.
    half_length = OBJECT_HALF_LENGTH
    self._env.sim.model.geom_rbound[env_ids, self._geom_id] = torch.sqrt(
      new_radius**2 + half_length**2
    )
    self._env.sim.model.geom_aabb[env_ids, self._geom_id, 1] = torch.stack(
      [new_radius, new_radius, torch.full_like(new_radius, half_length)], dim=-1
    )

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # The object itself is a real, visible geom (unlike the abstract point
    # targets of the other commands), so there is no hidden target to reveal.
    pass


@dataclass(kw_only=True)
class WrapCommandCfg(CommandTermCfg):
  object_entity_name: str = "object"
  object_geom_name: str = "object_geom"
  # Spawn region for the object CENTER, in the same base-relative polar
  # coordinates as the reach task (angle from +z, distance from the base).
  # Sign must stay OPPOSITE wrap_env_cfg.JOINT_RESET_RANGE: the object spawns
  # on the side the spirob leans AWAY from at reset (here, -x, while the reset
  # bias is >= 0 i.e. +x-leaning) -- see wrap_env_cfg.py's module docstring.
  #
  # This alone only keeps the CENTER clear; it's the object's SURFACE that
  # must not cross x=0 (into the reset pose) or dip below z=0 (through the
  # floor), so both bounds here are sized against object_radius_range's UPPER
  # end (0.10 m) with a margin, not against a point target. Tightening either
  # range without re-checking the other's corner cases can silently reopen
  # clipping -- verified empirically (min reset surface distance, min height
  # above floor, both > 0) across the full box, not just derived by hand.
  angle_range: tuple[float, float] = (-1.0, -0.6)
  center_radius_range: tuple[float, float] = (0.24, 0.32)
  # Cylinder radius. Lower bound comparable to the spirob's own segment
  # half-width at the tip (0.015 m); upper bound doubled from the original
  # 0.05 m on request, to also draw noticeably chunkier objects.
  object_radius_range: tuple[float, float] = (0.015, 0.10)
  success_threshold: float = 0.03

  def build(self, env: ManagerBasedRlEnv) -> WrapCommand:
    return WrapCommand(self, env)
