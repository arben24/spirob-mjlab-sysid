"""Shared entity handles and physical constants for the spirob task family."""

from __future__ import annotations

from pathlib import Path

from mjlab.managers.scene_entity_config import SceneEntityCfg
from spirob.paths import RL_MODEL

ENTITY_NAME = "spirob"

# The model the policies train on. Tracked in ``models/`` like every other
# MuJoCo model in this repository and resolved through ``spirob.paths`` rather
# than a path relative to this file -- see CLAUDE.md, "Paths".
SPIROB_XML: Path = RL_MODEL

# Measured table of statically holdable poses, used by the shape task to draw
# achievable targets. Regenerate with scripts/make_pose_table.py after changing
# the model or its dynamic parameters.
HOLDABLE_POSES_NPZ: Path = Path(__file__).parent.parent / "holdable_poses.npz"

TENDON_NAMES = ("tendon_0", "tendon_1")

# Segment chain runs seg_13 (at the base) down to seg_0 (the tip).
NUM_SEGMENTS = 14
TIP_SEGMENT = 0
# Segment whose site is used as the second tracked point in the shape task.
# 14 segments, so seg_7 sits one above the geometric middle of the chain.
MID_SEGMENT = 7

JOINTS_CFG = SceneEntityCfg(ENTITY_NAME, joint_names=("j_.*",))
TCP_CFG = SceneEntityCfg(ENTITY_NAME, site_names=("site_tcp",))
TENDONS_CFG = SceneEntityCfg(ENTITY_NAME, tendon_names=TENDON_NAMES)
IMU_CFG = SceneEntityCfg(
  ENTITY_NAME, site_names=tuple(f"site_imu_{i}" for i in range(NUM_SEGMENTS))
)
MID_SITE_CFG = SceneEntityCfg(ENTITY_NAME, site_names=(f"site_imu_{MID_SEGMENT}",))
BASE_BODY_CFG = SceneEntityCfg(ENTITY_NAME, body_names=("base",))

# Wrap task: the graspable object. A separate, jointless entity -- mjlab
# auto-wraps fixed-base entities with no free joint in a mocap body (see
# mjlab.utils.spec.auto_wrap_fixed_base_mocap), so its pose is set per env each
# reset via Entity.write_mocap_pose_to_sim rather than through the joint tree.
OBJECT_ENTITY_NAME = "object"
OBJECT_GEOM_NAME = "object_geom"
OBJECT_CFG = SceneEntityCfg(OBJECT_ENTITY_NAME)
OBJECT_GEOM_CFG = SceneEntityCfg(OBJECT_ENTITY_NAME, geom_names=(OBJECT_GEOM_NAME,))

# Matches every segment geom (g_0 .. g_13, one per segment body). Used by the
# wrap task's ContactSensor to track per-segment contact against the object.
SEGMENT_GEOM_PATTERN = r"^g_\d+$"
WRAP_CONTACT_SENSOR_NAME = "wrap_contact"

# Tendon length at the straight rest pose (all joints at zero), measured on the
# compiled model. Subtracted so the observation is centered around zero.
TENDON_REST_LEN = 0.4258

# Tendon force actuators: ctrlrange is (-150, 0), i.e. only pulling.
TENDON_FORCE_SCALE = 75.0
TENDON_FORCE_OFFSET = -75.0
TENDON_CTRL_RANGE = (-150.0, 0.0)

# Final DR draw widths reached at the end of the curriculum ramp -- also the
# widths used directly in DR-play (dr_in_play=True). lo < 1.0 means "let the
# value fall to lo * XML-default"; hi stays 1.0 so the XML value is the upper
# bound. Tune the lower bounds here. Terms not currently active are skipped.
DR_TARGETS = {
  "joint_stiffness": (0.5, 1.0),
  "joint_damping": (0.1, 1.0),
  "joint_friction": (0.1, 1.0),
  # "joint_armature": (0.6, 1.0),
  "joint_range": (0.9, 1.0),
  "tendon_stiffness": (0.5, 1.0),
  "tendon_damping": (0.5, 1.0),
  "tendon_frictionloss": (0.5, 1.0),
}
