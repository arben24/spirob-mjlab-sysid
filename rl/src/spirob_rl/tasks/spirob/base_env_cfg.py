"""Shared environment scaffolding for the spirob task family.

Everything that is *not* the task goal lives here: the entity, the tendon
action, the sensor-ablation observation ladder, domain randomization with its
curriculum, the grid layout, and the simulation settings. Each variant
(reach / shape / trajectory) supplies only its command and reward terms.
"""

from __future__ import annotations

from typing import Literal

import mujoco

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  dr,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: F401  (used by the commented-out segment_inertia DR term)
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import SensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .mdp.constants import (
  BASE_BODY_CFG,
  DR_TARGETS,
  ENTITY_NAME,
  IMU_CFG,
  JOINTS_CFG,
  SPIROB_XML,
  TCP_CFG,
  TENDON_CTRL_RANGE,
  TENDON_FORCE_OFFSET,
  TENDON_FORCE_SCALE,
  TENDON_NAMES,
  TENDONS_CFG,
)


# --- Domain-randomization switch (ablation) --------------------------------
# True  -> full DR: every term below is active and `dr_curriculum` widens it
#          from the no-op width toward DR_TARGETS during training.
# False -> no DR at all: the DR event terms and the curriculum are dropped, so
#          every environment runs at the nominal XML dynamics.
# Flip this to produce the with-DR / without-DR training pair; nothing else in
# the config needs to change.
ENABLE_DOMAIN_RANDOMIZATION = True

# Only read when ENABLE_DOMAIN_RANDOMIZATION is True.
# True  -> DR is phased in: every term starts at the no-op width (1.0, 1.0) and
#          `dr_curriculum` widens it toward DR_TARGETS over the first 5000
#          policy steps.
# False -> no ramp: every term sits at its DR_TARGETS width from step 0, so the
#          policy sees the full randomization for the whole run. The curriculum
#          term is dropped entirely.
ENABLE_DR_CURRICULUM = True


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(SPIROB_XML))


_SPIROB_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=TENDON_NAMES,
      transmission_type=TransmissionType.TENDON,
    ),
  ),
)


def get_spirob_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=_SPIROB_ARTICULATION,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )


SensorLevel = Literal["force", "tendon", "imu", "joints", "oracle"]
"""Which real-rig sensors the actor is allowed to see.

Ablation ladder, from least to most information. The critic always gets the
full privileged state, so a comparison across levels isolates the effect of
the actor's sensor suite.

* ``force``: only the tendon force. Note this equals the commanded action
  exactly (verified against ``tendonactuatorfrc``), so this level is
  effectively "no state feedback" and relies entirely on observation history.
* ``tendon``: adds the spool encoders (tendon length + velocity).
* ``imu``: adds the per-segment IMUs (segment inclination).
* ``joints``: the 13 joint angles and their velocities. Realizable on the rig
  via the accelerometer board (see ``rig/acc_board/``): each pair of
  adjacent sensors yields one joint angle, so the same ``joint_pos`` the sim
  computes directly is measurable on hardware. Unlike ``oracle`` this excludes
  ``tcp_pos``, which the rig cannot sense.
* ``oracle``: privileged joint angles, joint velocities and TCP position. Not
  realizable on the rig; upper bound for the comparison.
"""


def make_actor_terms(
  sensor_level: SensorLevel, command_name: str
) -> dict[str, ObservationTermCfg]:
  """Actor observations available at a given sensor level."""
  # Always present: the commanded tendon force (== last action) and the target.
  terms: dict[str, ObservationTermCfg] = {
    "target": ObservationTermCfg(
      func=generated_commands,
      params={"command_name": command_name},
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }

  if sensor_level in ("tendon", "imu"):
    terms["tendon_len"] = ObservationTermCfg(
      func=mdp.tendon_len_rel,
      params={"asset_cfg": TENDONS_CFG},
    )
    terms["tendon_vel"] = ObservationTermCfg(
      func=mdp.tendon_vel,
      params={"asset_cfg": TENDONS_CFG},
    )

  if sensor_level == "imu":
    terms["segment_pitch"] = ObservationTermCfg(
      func=mdp.segment_pitch_cos_sin,
      params={"asset_cfg": IMU_CFG},
    )

  # Joint angles are realizable on the rig (accelerometer board), so "joints"
  # and "oracle" share them; only "oracle" adds the unmeasurable TCP position.
  if sensor_level in ("joints", "oracle"):
    terms["joint_pos"] = ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": JOINTS_CFG},
    )
    terms["joint_vel"] = ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": JOINTS_CFG},
    )

  if sensor_level == "oracle":
    terms["tcp_pos"] = ObservationTermCfg(
      func=mdp.tcp_pos,
      params={"asset_cfg": TCP_CFG},
    )

  return terms


def _make_events(
  joint_reset_range: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, EventTermCfg]:
  events: dict[str, EventTermCfg] = {
    "reset_joints": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        # Each of the 13 joints draws its own independent offset in this
        # range (reset_joints_by_offset samples per-joint, not once for the
        # whole chain) -- e.g. (0.0, 0.5) biases every joint positive without
        # forcing them all to the same angle. See wrap_env_cfg.py for why the
        # wrap task narrows this to one side.
        "position_range": joint_reset_range,
        "velocity_range": (0.0, 0.0),
        "asset_cfg": JOINTS_CFG,
      },
    ),
    # Purely structural (not a difficulty knob like the DR terms below): lays
    # parallel envs out in a grid so multiple tentacles don't overlap when
    # watched together. Controlled by SceneCfg.env_spacing (--env.scene.env-spacing).
    "grid_layout": EventTermCfg(
      func=mdp.grid_layout,
      mode="startup",
      params={"asset_cfg": BASE_BODY_CFG},
    ),
  }

  # Domain randomization, curriculum-driven. Collected separately from the
  # structural events above so ENABLE_DOMAIN_RANDOMIZATION can drop the whole
  # block in one place. Each term uses operation="scale":
  # the field is multiplied per environment by a factor drawn from `ranges`,
  # applied to the XML default (not the current value, so it never accumulates).
  # The factors below all START at the no-op width (1.0, 1.0) -- i.e. exactly the
  # XML values, no spread -- because the `dr_curriculum` term further down
  # widens them toward their targets as training progresses (see DR_TARGETS).
  #
  # With scale, the XML value is automatically each joint's UPPER bound (factor
  # <= 1.0) and the draw reaches down to target_lo * default, so the per-segment
  # spread from the XML (damping runs 0.0001 .. 121) is preserved -- an absolute
  # range would collapse it. Comment a term out to disable it; the curriculum
  # skips whatever is not active.
  dr_events: dict[str, EventTermCfg] = {}
  dr_events["joint_stiffness"] = EventTermCfg(
    func=dr.joint_stiffness,
    mode="reset",
    params={"asset_cfg": JOINTS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  dr_events["joint_damping"] = EventTermCfg(
    func=dr.joint_damping,
    mode="reset",
    params={"asset_cfg": JOINTS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  dr_events["joint_friction"] = EventTermCfg(
    func=dr.joint_friction,
    mode="reset",
    params={"asset_cfg": JOINTS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  # dr_events["joint_armature"] = EventTermCfg(
  #   func=dr.joint_armature,
  #   mode="reset",
  #   params={"asset_cfg": JOINTS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  # )
  # dr_events["joint_range"] = EventTermCfg(
  #   func=dr.joint_limits,
  #   mode="reset",
  #   params={"asset_cfg": JOINTS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  # )
  #
  # Tendon properties -- these sit directly in the actuation path, so they tend
  # to matter more for sim-to-real than the joint terms above.
  dr_events["tendon_stiffness"] = EventTermCfg(
    func=dr.tendon_stiffness,
    mode="reset",
    params={"asset_cfg": TENDONS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  dr_events["tendon_damping"] = EventTermCfg(
    func=dr.tendon_damping,
    mode="reset",
    params={"asset_cfg": TENDONS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  dr_events["tendon_frictionloss"] = EventTermCfg(
    func=dr.tendon_frictionloss,
    mode="reset",
    params={"asset_cfg": TENDONS_CFG, "ranges": (1.0, 1.0), "operation": "scale"},
  )
  #
  # Terms that do NOT fit the scale curriculum above (different no-op width or
  # param name), so they are not managed by dr_curriculum:
  #   joint_play  -- assembly play ("Los"). MuJoCo has no true backlash param (a
  #     dead zone would need an extra DOF per joint); this shifts each joint's
  #     zero point instead. operation="add", so its curriculum no-op is (0, 0).
  #   segment_inertia -- takes alpha_range, not ranges.
  # dr_events["joint_play"] = EventTermCfg(
  #   func=dr.joint_default_pos,
  #   mode="reset",
  #   params={"asset_cfg": JOINTS_CFG, "ranges": (0.0, 0.0), "operation": "add"},
  # )
  # dr_events["segment_inertia"] = EventTermCfg(
  #   func=dr.pseudo_inertia,
  #   mode="reset",
  #   params={
  #     "asset_cfg": SceneEntityCfg(ENTITY_NAME, body_names=("seg_.*",)),
  #     "alpha_range": (-0.1, 0.1),
  #   },
  # )

  if ENABLE_DOMAIN_RANDOMIZATION:
    if not ENABLE_DR_CURRICULUM:
      # Without the ramp the terms would stay at their no-op width forever (the
      # curriculum is what moves them), so pin each one to its final width here.
      for name, term in dr_events.items():
        if name in DR_TARGETS:
          term.params["ranges"] = DR_TARGETS[name]
    events.update(dr_events)
  return events


def make_base_env_cfg(
  *,
  commands: dict[str, CommandTermCfg],
  rewards: dict[str, RewardTermCfg],
  command_name: str,
  sensor_level: SensorLevel = "tendon",
  history_length: int = 10,
  episode_length_s: float = 50.0,
  extra_critic_terms: dict[str, ObservationTermCfg] | None = None,
  extra_entities: dict[str, EntityCfg] | None = None,
  extra_sensors: tuple[SensorCfg, ...] = (),
  joint_reset_range: tuple[float, float] = (-0.5, 0.5),
) -> ManagerBasedRlEnvCfg:
  """Assemble a spirob env config around a variant's commands and rewards."""
  actor_terms = make_actor_terms(sensor_level, command_name)

  # Critic always sees the full state, independent of the actor's level.
  critic_terms: dict[str, ObservationTermCfg] = {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": JOINTS_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": JOINTS_CFG},
    ),
    "tcp_pos": ObservationTermCfg(
      func=mdp.tcp_pos,
      params={"asset_cfg": TCP_CFG},
    ),
    "target": ObservationTermCfg(
      func=generated_commands,
      params={"command_name": command_name},
    ),
    "last_action": ObservationTermCfg(func=last_action),
  }
  if extra_critic_terms:
    critic_terms.update(extra_critic_terms)

  observations = {
    # History gives the actor the state information a single encoder reading
    # cannot carry (the tentacle shape is not uniquely determined by one
    # tendon-length sample).
    "actor": ObservationGroupCfg(
      actor_terms, enable_corruption=True, history_length=history_length
    ),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    # Policy action in [-1, 1] maps to a pulling force in [-150, 0] N per tendon.
    "tendon_force": TendonEffortActionCfg(
      entity_name=ENTITY_NAME,
      actuator_names=TENDON_NAMES,
      scale=TENDON_FORCE_SCALE,
      offset=TENDON_FORCE_OFFSET,
      clip={".*": TENDON_CTRL_RANGE},
    ),
  }

  # end_step is in policy steps: max_iterations * num_steps_per_env. Without DR
  # there is nothing to ramp, so the curriculum is empty.
  curriculum: dict[str, CurriculumTermCfg] = {}
  if ENABLE_DOMAIN_RANDOMIZATION and ENABLE_DR_CURRICULUM:
    curriculum["dr_curriculum"] = CurriculumTermCfg(
      func=mdp.dr_range_curriculum,
      params={
        "targets": DR_TARGETS,
        "start_step": 0,
        "end_step": 5000,
        "start_ranges": (1.0, 1.0),
      },
    )

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  entities = {ENTITY_NAME: get_spirob_cfg()}
  if extra_entities:
    entities.update(extra_entities)

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      # The spirob XML brings its own floor plane, so no scene terrain.
      entities=entities,
      sensors=extra_sensors,
      num_envs=10,
      env_spacing=0.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=_make_events(joint_reset_range=joint_reset_range),
    curriculum=curriculum,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name=ENTITY_NAME,
      body_name="base",
      distance=0.8,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      # Solver settings taken from the <option> element of spirob.xml.
      mujoco=MujocoCfg(
        timestep=0.004,
        impratio=18.7818,
        cone="elliptic",
        iterations=20,
      ),
    ),
    decimation=5,  # 50 Hz policy at 0.004 s sim timestep.
    episode_length_s=episode_length_s,
  )


def apply_play_overrides(
  cfg: ManagerBasedRlEnvCfg, dr_in_play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Turn a training config into a play config.

  Shared by every variant so DR-play behaves identically across tasks.
  """
  cfg.observations["actor"].enable_corruption = False
  # The curriculum is meaningless in play (common_step_counter starts at 0, so
  # it would just hold every range at the nominal no-op width). Drop it either
  # way and decide the physics below.
  cfg.curriculum = {}
  if dr_in_play:
    # Watch the trained policy under randomized dynamics: pin every active DR
    # term to its full-width target (what the curriculum reaches at the end of
    # training) instead of ramping from zero. Shorter episodes so each reset
    # draws fresh dynamics; use --num-envs > 1 to see several draws at once.
    cfg.episode_length_s = 20.0
    cfg.scene.num_envs = 4
    for name, term in cfg.events.items():
      if name in DR_TARGETS and "operation" in term.params:
        term.params["ranges"] = DR_TARGETS[name]
  else:
    # Evaluate at the nominal XML values: drop every scale/abs DR event term
    # so nothing perturbs the dynamics.
    cfg.episode_length_s = 1e10
    cfg.events = {
      name: term
      for name, term in cfg.events.items()
      if "operation" not in term.params
    }
  return cfg


def make_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  """PPO settings shared by every spirob variant."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 256, 128, 128), # 128,128,128,128,64
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 2.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128,128,128,128,64), # 128,128,128,128,64
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.3,
      entropy_coef=0.05,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=128,
    max_iterations=500,
    # Hard bound on the raw (pre-scale) action actually applied to the env,
    # enforced by RslRlVecEnvWrapper before every step -- unlike the
    # TendonEffortAction's own clip (which only bounds the physical ctrl
    # after scale/offset), this keeps env.action_manager.action itself finite.
    # Without it, action_rate_l2/action_l2 compute on an unbounded raw action:
    # a bad gradient step can walk the policy mean out past the actuator's own
    # saturation point, where nothing in the physics pulls it back, and the
    # quadratic penalty on an ever-growing raw action explodes the value loss
    # to inf within a couple of iterations. Observed exactly this way on the
    # wrap task once domain randomization reached full width (more contact,
    # less damping -> larger, noisier policy gradients): stable through
    # iteration ~90, "Mean value loss: inf" and a crash by iteration 100.
    # 5.0 is generous headroom over the useful range (the mapped ctrl already
    # saturates at |raw| ~= 1) without constraining exploration noise.
    clip_actions=5.0,
  )
