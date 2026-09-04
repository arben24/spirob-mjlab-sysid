"""Reachable-workspace geometry for the SpiRob TCP reaching task.

Mirrors ``TcpPositionCommandCfg``'s target sampling in
``spirob_rl/tasks/spirob/reach_env_cfg.py`` -- keep the four
constants below in sync if those defaults change.

Deliberately stdlib-only: ``target_gui.py`` imports this, and pulling in torch
plus the whole task registry just to draw an arc would make the GUI slow to
start and couple it to the training stack.

The tentacle is inextensible, so reachable TCP positions do not fill an area --
they form a thin arc-shaped shell. Bending sideways trades tip angle for
radius:

    r(angle) = SHELL_RADIUS - SHELL_CURVATURE * angle**2   (+- SHELL_BAND)

with ``angle`` measured from the +z axis (positive toward +x) and limited to
``+-ANGLE_LIMIT``. Everything is in metres / radians in the robot base frame,
where x is sideways and z is up.
"""

from __future__ import annotations

import math
from typing import NamedTuple

ANGLE_LIMIT = 1.7
SHELL_RADIUS = 0.33
SHELL_CURVATURE = 0.045
SHELL_BAND = 0.08

TARGET_UDP_PORT = 47800
"""Default localhost UDP port the target GUI pushes setpoints to.

Datagrams are plain ASCII ``"x z"`` -- the same text format the bridge accepts
on stdin, so both input paths share one parser.
"""


class TargetInfo(NamedTuple):
  angle: float
  """Angle from the +z axis [rad], positive toward +x."""
  radius: float
  """Distance from the base [m]."""
  shell_radius: float
  """Shell radius at this angle [m]."""
  deviation: float
  """Signed distance from the shell [m]; + is beyond it, - is inside it."""
  angle_ok: bool
  radius_ok: bool

  @property
  def ok(self) -> bool:
    """True if the target is inside the distribution the policy was trained on."""
    return self.angle_ok and self.radius_ok


def shell_radius(angle: float) -> float:
  return SHELL_RADIUS - SHELL_CURVATURE * angle**2


def shell_point(angle: float) -> tuple[float, float]:
  """Point (x, z) on the shell centerline at the given angle."""
  radius = shell_radius(angle)
  return radius * math.sin(angle), radius * math.cos(angle)


def describe(x: float, z: float) -> TargetInfo:
  radius = math.hypot(x, z)
  angle = math.atan2(x, z)
  r_shell = shell_radius(angle)
  deviation = radius - r_shell
  return TargetInfo(
    angle=angle,
    radius=radius,
    shell_radius=r_shell,
    deviation=deviation,
    angle_ok=abs(angle) <= ANGLE_LIMIT,
    radius_ok=abs(deviation) <= SHELL_BAND,
  )


def snap_to_shell(x: float, z: float) -> tuple[float, float]:
  """Nearest point on the shell centerline, clamped to the trained angle range."""
  angle = max(-ANGLE_LIMIT, min(ANGLE_LIMIT, math.atan2(x, z)))
  return shell_point(angle)


def warnings_for(x: float, z: float) -> list[str]:
  """Human-readable reasons why (x, z) is outside the trained workspace.

  Empty list means the target is fine. Callers decide how loudly to report --
  the bridge only prints on the transition in or out, so a dragged GUI slider
  cannot flood the console.
  """
  info = describe(x, z)
  messages: list[str] = []
  if not info.angle_ok:
    messages.append(
      f"target angle {info.angle:+.2f} rad is outside the trained range "
      f"+-{ANGLE_LIMIT:.2f} rad -- policy is extrapolating"
    )
  if not info.radius_ok:
    messages.append(
      f"target radius {info.radius:.3f} m is {info.deviation:+.3f} m off the trained "
      f"shell ({info.shell_radius:.3f} +-{SHELL_BAND:.2f} m at angle "
      f"{info.angle:+.2f} rad) -- the tentacle is inextensible, this point is "
      "likely not holdable"
    )
  return messages


def parse_target(text: str) -> tuple[float, float] | None:
  """Parse an ``"x z"`` line into a target, or None if it is not one.

  Shared by the bridge's stdin reader and its UDP socket so both accept
  exactly the same format.
  """
  parts = text.split()
  if len(parts) != 2:
    return None
  try:
    return float(parts[0]), float(parts[1])
  except ValueError:
    return None
