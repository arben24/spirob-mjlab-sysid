"""MJCF spec for the wrap task's graspable cylinder object.

A standalone entity, not part of the spirob's own body tree: a single body with
one cylinder geom and no joint. mjlab auto-wraps fixed-base entities with no
free joint in a mocap body (see ``mjlab.utils.spec.auto_wrap_fixed_base_mocap``),
which is what lets :class:`~.commands.WrapCommand` place it anywhere per
environment via ``write_mocap_pose_to_sim`` instead of it being pinned to the
world origin.

The geom uses ``fromto`` so its length runs along the body's local y axis
regardless of the body's own orientation -- perpendicular to the spirob's x-z
bending plane, like a horizontal bar the tentacle coils around in that plane.

Collision is on: contype/conaffinity/friction/solref/solimp are left at
MuJoCo's defaults, matching the spirob's own segment geoms (spirob.xml sets
none of these per-geom either). The object is a mocap body, so it is
kinematically placed rather than dynamically simulated -- it does not get
pushed by contact forces, staying put like a pole bolted in place -- but it
fully participates in collision: the spirob's segments generate real normal
and friction forces against it and cannot pass through it.
"""

from __future__ import annotations

import mujoco

from .constants import OBJECT_GEOM_NAME

# Half-length of the cylinder along y. Fixed (not randomized), wide enough to
# clear the spirob segments' own y half-extent (~0.046 m at the base) so a
# segment can't slip past either end of the cylinder while wrapping.
OBJECT_HALF_LENGTH = 0.08

# Compile-time default radius. Overwritten per environment by WrapCommand on
# every reset, so this value only matters before the first reset.
_DEFAULT_RADIUS = 0.03


def get_object_spec(
  radius: float = _DEFAULT_RADIUS,
  half_length: float = OBJECT_HALF_LENGTH,
  rgba: tuple[float, float, float, float] = (0.9, 0.35, 0.1, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_geom(
    name=OBJECT_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(radius, half_length, 0.0),
    fromto=(0.0, -half_length, 0.0, 0.0, half_length, 0.0),
    rgba=rgba,
  )
  return spec
