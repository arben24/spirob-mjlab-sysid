"""Planar forward kinematics of the spirob segment chain.

Used by the shape task to sample *kinematically consistent* target points: two
independently drawn points would almost never lie on one achievable tentacle
pose, so instead a joint configuration is drawn and both targets are read off
its forward kinematics.

All motion is planar (every hinge turns about the y axis), so a pose reduces to
one angle per joint and the chain can be walked analytically in torch, batched
over environments -- no MuJoCo call per sample.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch

from .constants import NUM_SEGMENTS


class PlanarChain:
  """Batched planar FK for the seg_13 (base) .. seg_0 (tip) chain.

  Geometry is read once from the nominal compiled model, so it reflects the XML
  rather than any per-world grid offset or domain randomization applied later.
  """

  def __init__(self, xml_path: Path, device: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.device = device

    def body_id(name: str) -> int:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def site_id(name: str) -> int:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    # Offset from seg_{s+1} to seg_s, ordered base -> tip (13 entries).
    offsets = [
      float(model.body_pos[body_id(f"seg_{s}")][2])
      for s in range(NUM_SEGMENTS - 2, -1, -1)
    ]
    self.link_offsets = torch.tensor(offsets, dtype=torch.float32, device=device)

    # Local z offset of each segment's IMU site, indexed by segment number.
    self.site_offsets = torch.tensor(
      [float(model.site_pos[site_id(f"site_imu_{s}")][2]) for s in range(NUM_SEGMENTS)],
      dtype=torch.float32,
      device=device,
    )
    self.tcp_offset = float(model.site_pos[site_id("site_tcp")][2])

    self.num_joints = model.njnt
    self.joint_limit = float(model.jnt_range[0][1])

  def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk the chain for a batch of joint configurations.

    Args:
      q: Joint angles ``[N, 13]`` in chain order (j_12 at the base first). This
        matches the MuJoCo joint ordering of the model.

    Returns:
      ``(origins, thetas)`` where ``origins`` is ``[N, 14, 3]`` holding each
      segment's frame origin (y is always 0) and ``thetas`` is ``[N, 14]``
      holding its accumulated pitch. Both are indexed by segment number, so
      index 13 is the base segment and index 0 the tip.
    """
    n = q.shape[0]
    origins = torch.zeros(n, NUM_SEGMENTS, 3, device=q.device, dtype=q.dtype)
    thetas = torch.zeros(n, NUM_SEGMENTS, device=q.device, dtype=q.dtype)

    x = torch.zeros(n, device=q.device, dtype=q.dtype)
    z = torch.zeros(n, device=q.device, dtype=q.dtype)
    theta = torch.zeros(n, device=q.device, dtype=q.dtype)

    # Segment 13 sits at the base with zero pitch; entries are already zero.
    for i, seg in enumerate(range(NUM_SEGMENTS - 2, -1, -1)):
      length = self.link_offsets[i]
      x = x + length * torch.sin(theta)
      z = z + length * torch.cos(theta)
      theta = theta + q[:, i]
      origins[:, seg, 0] = x
      origins[:, seg, 2] = z
      thetas[:, seg] = theta

    return origins, thetas

  def site_pos(
    self, origins: torch.Tensor, thetas: torch.Tensor, segment: int
  ) -> torch.Tensor:
    """Position of ``site_imu_<segment>`` given a ``forward`` result. ``[N, 3]``."""
    offset = self.site_offsets[segment]
    theta = thetas[:, segment]
    pos = origins[:, segment].clone()
    pos[:, 0] = pos[:, 0] + offset * torch.sin(theta)
    pos[:, 2] = pos[:, 2] + offset * torch.cos(theta)
    return pos

  def tcp_pos(self, origins: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Position of ``site_tcp`` given a ``forward`` result. ``[N, 3]``."""
    theta = thetas[:, 0]
    pos = origins[:, 0].clone()
    pos[:, 0] = pos[:, 0] + self.tcp_offset * torch.sin(theta)
    pos[:, 2] = pos[:, 2] + self.tcp_offset * torch.cos(theta)
    return pos

class HoldablePoseTable:
  """Joint configurations the tendon-driven spirob can actually hold.

  With 13 joints but only 2 tendon forces, the statically holdable poses form
  at most a 2-manifold in joint space -- the image of the control square under
  the quasi-static map. Analytic guesses at that manifold fit badly (a linear
  curvature+taper family leaves ~0.2 rad residual per joint, against a 0.51 rad
  joint limit), so it is measured instead: ``_make_pose_table.py`` sweeps a grid
  over the two tendon commands, lets each settle, and records the resulting
  pose.

  Sampling draws continuous coordinates inside that control grid and bilinearly
  interpolates, so coverage is continuous rather than a few hundred fixed poses.
  Neighbouring grid cells are close in control space, which is what makes the
  interpolated poses near-achievable too.

  Note the recorded pose is the mean over the last second of settling: the base
  joint is almost undamped in the XML (1e-4), so some poses keep ringing with a
  small amplitude (~0.011 rad mean peak-to-peak) rather than coming fully to
  rest. Targets are therefore the pose the tentacle oscillates *about*.
  """

  def __init__(self, path: Path, device: str) -> None:
    with np.load(path) as data:
      table = data["joint_pos"]
    self.table = torch.tensor(table, dtype=torch.float32, device=device)
    self.grid = self.table.shape[0]
    self.device = device

  def sample(self, n: int, jitter: float = 0.0) -> torch.Tensor:
    """Draw ``n`` holdable joint configurations. ``[n, num_joints]``."""
    uv = torch.rand(n, 2, device=self.device) * (self.grid - 1)
    i0 = uv[:, 0].floor().long().clamp(0, self.grid - 2)
    j0 = uv[:, 1].floor().long().clamp(0, self.grid - 2)
    fi = (uv[:, 0] - i0).unsqueeze(-1)
    fj = (uv[:, 1] - j0).unsqueeze(-1)

    q = (
      self.table[i0, j0] * (1 - fi) * (1 - fj)
      + self.table[i0 + 1, j0] * fi * (1 - fj)
      + self.table[i0, j0 + 1] * (1 - fi) * fj
      + self.table[i0 + 1, j0 + 1] * fi * fj
    )
    if jitter > 0.0:
      q = q + torch.empty_like(q).uniform_(-jitter, jitter)
    return q
