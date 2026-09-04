"""Spirob wrap task: coil around a randomly placed, randomly sized cylinder.

A cylinder spawns in a defined region on one side of the spirob, with a random
radius, and the policy has to wrap the tentacle around it -- not just touch it
with the tip, but bring segments along the whole chain into contact with its
surface, spread around it rather than clustered on one side.

The reset is biased *away* from the object, on purpose: every joint
independently draws a non-negative offset from ``JOINT_RESET_RANGE`` (see
``base_env_cfg.make_base_env_cfg``'s ``joint_reset_range``), while the object
spawns at negative x (``WrapCommandCfg.angle_range`` in ``mdp/commands.py`` is
negative). "Positive joint angle" bends the chain toward +x per the forward
kinematics in ``mdp/kinematics.py`` (``x = length * sin(theta)``), so with
every joint angle >= 0 the accumulated pitch never goes negative either --
the reset pose geometrically cannot reach into -x, where the object sits. That
rules out spawning clipped into the cylinder, not just makes it unlikely. The
two ranges are opposite-sign ON PURPOSE and live in different files; keep them
opposite (not equal/matching) if you retune either one.

Object and joints are always resampled together, and *only* at an actual
environment reset: every reset goes through
``ManagerBasedRlEnv._reset_idx``, which runs the ``reset_joints`` event and
then unconditionally calls ``command_manager.reset()`` for every active
command (mjlab's ``CommandTerm.reset()`` force-resamples regardless of its own
``resampling_time_range`` countdown). ``WrapCommandCfg.resampling_time_range``
is set far above any episode length specifically so it never *also* fires
mid-episode the way it does for reach/shape/trajectory's abstract point
targets: those have no physical presence, so a new target mid-episode is
harmless, but the object here is a real collidable body -- resampling it while
the spirob is mid-motion, still wherever the policy currently has it, would
place a new cylinder without regard for the tentacle's current pose and could
spawn it overlapping.

The object has real collision: contype/conaffinity/friction/solref/solimp all
default (see mdp/object_spec.py), matching the spirob's own segment geoms. It
is a mocap body, so it stays fixed in place -- like a pole bolted down -- but
the spirob's segments generate real contact forces against it and cannot pass
through it.

Reward (see mdp/rewards.py):

* ``wrap_proximity`` / ``wrap_proximity_fine`` -- how close do segments get to
  the object's surface (purely geometric).
* ``wrap_coverage`` -- how well the close segments spread around it, via
  circular-statistics resultant length (also geometric).
* ``wrap_force_distribution`` -- unlike the three above, this one reads *real*
  per-segment contact force off a :class:`~mjlab.sensor.ContactSensor` between
  the spirob's segment geoms and the object (``WRAP_CONTACT_SENSOR_NAME``
  below). Geometric closeness says a segment is touching the surface, not how
  hard, so a policy could satisfy proximity/coverage with twelve segments
  barely grazing it and one doing all the gripping; this term is what actually
  penalizes that.
"""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import action_rate_l2, joint_vel_l2  # noqa: F401  (commented-out reward terms below)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from . import mdp
from .base_env_cfg import (
  SensorLevel,
  apply_play_overrides,
  make_base_env_cfg,
  make_ppo_runner_cfg,
)
from .mdp.constants import (
  ENTITY_NAME,
  JOINTS_CFG,  # noqa: F401  (commented-out joint_vel reward term below)
  OBJECT_ENTITY_NAME,
  OBJECT_GEOM_NAME,
  SEGMENT_GEOM_PATTERN,
  WRAP_CONTACT_SENSOR_NAME,
)

COMMAND_NAME = "wrap_target"

# Every joint's reset offset is drawn from this range independently (see
# base_env_cfg._make_events); (0.0, 0.5) keeps every joint non-negative, so the
# spirob always starts leaning toward +x -- away from the object, which spawns
# at negative x (WrapCommandCfg.angle_range in mdp/commands.py). Opposite sign
# on purpose: see the module docstring for why.
JOINT_RESET_RANGE = (0.0, 0.5)


def make_env_cfg(
  sensor_level: SensorLevel = "tendon",
  history_length: int = 10,
) -> ManagerBasedRlEnvCfg:
  commands = {
    COMMAND_NAME: mdp.WrapCommandCfg(
      # Deliberately never fires mid-episode (see resampling_time_range's
      # docstring: "governs extra resamples within an episode"). Unlike the
      # abstract point targets in reach/shape/trajectory, the object is a real
      # collidable body -- a mid-episode resample would teleport it to a new
      # position while the spirob stays wherever the policy currently has it,
      # which can materialize the cylinder overlapping the tentacle (clipping,
      # or the spirob ending up pinned underneath it). Only the guaranteed
      # reset-time resample is safe (joints reset in the same _reset_idx call,
      # verified clip-free), so this is set comfortably above any
      # episode_length_s this task uses, not tied to the current value of it.
      resampling_time_range=(1.0e6, 1.0e6),
      debug_vis=True,
    ),
  }

  rewards = {
    # Coarse: get segments generally near the object's surface.
    "wrap_proximity": RewardTermCfg(
      func=mdp.wrap_proximity,
      weight=1.0,
      params={"std": 0.08, "command_name": COMMAND_NAME},
    ),
    # Fine: hug the surface precisely.
    "wrap_proximity_fine": RewardTermCfg(
      func=mdp.wrap_proximity,
      weight=1.0,
      params={"std": 0.03, "command_name": COMMAND_NAME},
    ),
    # Spread around the object rather than clustering on one side. Weighted
    # lower than proximity: it should refine an already-close wrap, not pull
    # segments toward a "spread out" configuration that isn't near the object.
    "wrap_coverage": RewardTermCfg(
      func=mdp.wrap_coverage,
      weight=1.5,
      params={"proximity_std": 0.05, "command_name": COMMAND_NAME},
    ),
    # Real contact force spread evenly across segments, not just their
    # positions -- see the module docstring and mdp/rewards.py for why this is
    # a separate term from wrap_coverage. Weighted lower still: it should only
    # matter once there is real force to distribute in the first place, which
    # proximity/coverage are what actually create.
    "wrap_force_distribution": RewardTermCfg(
      func=mdp.wrap_force_distribution,
      weight=1.0,
      params={"sensor_name": WRAP_CONTACT_SENSOR_NAME},
    ),
    #"action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    # Keeps the policy mean inside the saturating action range. See action_l2.
    #"action_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.002),
    # "joint_vel": RewardTermCfg(
    #   func=joint_vel_l2,
    #   weight=-0.005,
    #   params={"asset_cfg": JOINTS_CFG},
    # ),
  }

  extra_entities: dict[str, EntityCfg] = {
    OBJECT_ENTITY_NAME: EntityCfg(spec_fn=mdp.get_object_spec),
  }

  # Per-segment contact force against the object, for wrap_force_distribution.
  # reduce="netforce" sums every raw contact on a segment into one net wrench
  # (a box-vs-cylinder pair could in principle produce more than one contact
  # point at corners/edges) rather than "maxforce", which would only report
  # the strongest and silently drop the rest of that segment's actual load.
  extra_sensors = (
    ContactSensorCfg(
      name=WRAP_CONTACT_SENSOR_NAME,
      primary=ContactMatch(
        mode="geom", pattern=SEGMENT_GEOM_PATTERN, entity=ENTITY_NAME
      ),
      secondary=ContactMatch(
        mode="geom", pattern=OBJECT_GEOM_NAME, entity=OBJECT_ENTITY_NAME
      ),
      fields=("force",),
      reduce="netforce",
    ),
  )

  cfg = make_base_env_cfg(
    commands=commands,
    rewards=rewards,
    command_name=COMMAND_NAME,
    sensor_level=sensor_level,
    history_length=history_length,
    episode_length_s=30.0,
    extra_entities=extra_entities,
    extra_sensors=extra_sensors,
    joint_reset_range=JOINT_RESET_RANGE,
  )
  # A tight wrap can put many segments in simultaneous contact with the
  # cylinder (plus the spirob's own pre-existing self-contact), well past what
  # the reach/shape/trajectory tasks ever generate. The heuristic default
  # (nconmax/njmax=None) overflowed under that load ("nefc overflow" warnings,
  # silently dropped constraints) and training on this task alone diverged
  # within ~60 iterations. Sized generously above what was observed during a
  # 2000-step random-action stress test (peak actual usage well under 100).
  cfg.sim.nconmax = 300
  cfg.sim.njmax = 400
  return cfg


def wrap_env_cfg(
  play: bool = False,
  sensor_level: SensorLevel = "tendon",
  history_length: int = 5,
  dr_in_play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_env_cfg(sensor_level=sensor_level, history_length=history_length)
  if play:
    apply_play_overrides(cfg, dr_in_play=dr_in_play)
  return cfg


def wrap_ppo_runner_cfg(
  sensor_level: SensorLevel = "tendon",
) -> RslRlOnPolicyRunnerCfg:
  return make_ppo_runner_cfg(f"rl_explor_spirob_wrap_{sensor_level}")
