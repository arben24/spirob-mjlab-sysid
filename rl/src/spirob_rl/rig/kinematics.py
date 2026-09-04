"""Forward kinematics of the SpiRob, driven by the accelerometer board.

The rig can measure its joint angles (``rig/acc_board``, 14 accelerometers
along the chain) but not where its tip actually is. Since the tentacle is a
rigid-link chain with known segment lengths, the tip follows from the angles --
so the one quantity the task is rewarded on, ``|TCP - target|``, becomes
measurable on hardware after all. That is what this module is for: the bridge
uses it to report where the arm *is* against where it was *told* to go.

Two conversions live here, and both matter for the hardware:

* **Segment inclination.** The sim's ``segment_pitch`` observation is the
  cos/sin of each segment's absolute pitch, and that is exactly the cumulative
  sum of the joint angles from the base: ``pitch[s] = sum(q[:s])``. Verified
  against the compiled model to machine precision in ``tests/``. This is what
  makes the ``imu`` sensor level realizable -- the accelerometers measure
  inclination directly, and their differences are what the joint angles were
  computed from, so summing them back gives the same numbers.
* **Tip position.** Planar chain, all hinges about y, so the tip is a sum of
  rotated segment lengths.

The segment lengths come from the same MJCF the task trains on, read at
runtime. A copied constant is a constant that silently drifts.

Deliberately not imported by ``target_gui.py``: the GUI is stdlib-only so it
starts instantly and cannot be dragged into the training stack. It receives the
finished tip position from the bridge over UDP instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from spirob.paths import RL_MODEL

_DEFAULT_XML = RL_MODEL


@dataclass(frozen=True)
class ChainGeometry:
  """The rigid part of the robot: what does not change while it bends."""

  link_lengths: tuple[float, ...]
  """Offset from each segment's origin to the next one, base->tip [m].

  One entry per segment; the last is the offset from the tip segment's origin
  to the TCP site, so ``sum(link_lengths)`` is the fully extended reach.
  """
  joint_names: tuple[str, ...]
  """Sim joint names base->tip (``j_12`` .. ``j_0``)."""
  joint_limits: tuple[float, ...]
  """Per-joint symmetric limit [rad], as compiled from the MJCF."""

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def num_segments(self) -> int:
    return len(self.link_lengths)

  @property
  def total_length(self) -> float:
    return float(sum(self.link_lengths))


@lru_cache(maxsize=4)
def load_chain_geometry(xml_path: str | None = None) -> ChainGeometry:
  """Read segment lengths, joint order and limits from the task's MJCF.

  Compiled through MuJoCo rather than parsed as text: the compiler is the
  authority on what the model actually is, and this runs once at startup.
  """
  import mujoco  # local: keeps the import cost off callers that never need it

  path = Path(xml_path) if xml_path else _DEFAULT_XML
  model = mujoco.MjModel.from_xml_path(str(path))

  def body_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

  # Walk base->tip along seg_13 .. seg_0, exactly the chain in the XML.
  segment_names = [f"seg_{i}" for i in range(13, -1, -1)]
  ids = [body_id(name) for name in segment_names]
  if any(i < 0 for i in ids):
    missing = [n for n, i in zip(segment_names, ids) if i < 0]
    raise ValueError(f"{path} does not contain the expected chain bodies: {missing}")

  # A child body's pos is the offset from its parent's origin, and every joint
  # sits at its own body's origin -- so this offset is the link length.
  lengths = [float(model.body_pos[child][2]) for child in ids[1:]]
  tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "site_tcp")
  if tcp_site < 0:
    raise ValueError(f"{path} has no 'site_tcp' -- cannot locate the TCP.")
  lengths.append(float(model.site_pos[tcp_site][2]))

  joint_names: list[str] = []
  joint_limits: list[float] = []
  for body in ids[1:]:  # seg_13 is fixed to the base and carries no joint
    for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
      joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
      lo, hi = model.jnt_range[j]
      joint_limits.append(float(max(abs(lo), abs(hi))))

  return ChainGeometry(
    link_lengths=tuple(lengths),
    joint_names=tuple(joint_names),
    joint_limits=tuple(joint_limits),
  )


def segment_pitch(joint_pos: np.ndarray | list[float]) -> np.ndarray:
  """Absolute inclination of every segment, base->tip. Shape (num_joints + 1,).

  The base segment is welded to the mount, so its pitch is zero by definition
  and every further segment adds one joint angle. On hardware the chain runs
  the other way -- the accelerometers read inclinations and their differences
  are the joint angles -- which is why the two agree without any extra
  reference: summing telescopes back to exactly what was differenced, and the
  gravity reference cancels with it.
  """
  q = np.asarray(joint_pos, dtype=float)
  return np.concatenate([[0.0], np.cumsum(q)])


def segment_pitch_cos_sin(joint_pos: np.ndarray | list[float]) -> list[float]:
  """The sim's ``segment_pitch`` observation, rebuilt from joint angles.

  Layout must match ``segment_pitch_cos_sin`` in the task config: all cosines
  base->tip, then all sines. The sim resolves its IMU sites in model order,
  which runs ``site_imu_13`` (base) to ``site_imu_0`` (tip) -- the same
  direction the accelerometer chain is read in, so nothing has to be reordered.
  """
  pitch = segment_pitch(joint_pos)
  return [*np.cos(pitch), *np.sin(pitch)]


def forward_kinematics(q: np.ndarray | list[float], geometry: ChainGeometry) -> np.ndarray:
  """Joint origins along the chain, base->tip, plus the TCP. Shape (N+1, 2).

  Planar chain in the x-z plane: a rotation by ``theta`` about +y sends the
  segment's local +z to ``(sin theta, cos theta)``, which is the same
  convention ``workspace.shell_point`` uses for targets.
  """
  q = np.asarray(q, dtype=float)
  if q.shape != (geometry.num_joints,):
    raise ValueError(f"expected {geometry.num_joints} joint angles, got {q.shape}")

  points = np.zeros((geometry.num_segments + 1, 2))
  theta = 0.0
  x = z = 0.0
  for s, length in enumerate(geometry.link_lengths):
    if s > 0:  # segment 0 is welded to the base; joint s-1 sits at segment s
      theta += float(q[s - 1])
    x += length * math.sin(theta)
    z += length * math.cos(theta)
    points[s + 1] = (x, z)
  return points


def tcp_position(q: np.ndarray | list[float], geometry: ChainGeometry) -> tuple[float, float]:
  """TCP (x, z) in metres in the robot base frame."""
  tip = forward_kinematics(q, geometry)[-1]
  return float(tip[0]), float(tip[1])
