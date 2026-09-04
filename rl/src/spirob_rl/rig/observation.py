"""Checkpoint-derived observation assembly for the SpiRob policy bridge.

The policy is the source of truth for what it must be fed. Instead of a
hand-maintained table of "sensor level -> terms" (which silently rots when the
env config changes), this module builds the task's env once, reads the actor
observation layout straight off mjlab's observation manager -- term order, each
term's pre-history dimension, and the history length -- and reconstructs that
exact vector at runtime from live hardware signals.

The bridge then only has to satisfy each term. A small registry says where each
term's data comes from: some are internal (the TCP target, the last action),
the rest map to a named signal a hardware source provides (``tendon_len``,
``joint_pos``, ``segment_pitch``, ...). A term with no known origin
(``tcp_pos``) is flagged as not realizable, with a clear message, before any
motor moves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

# The observation term that carries the commanded TCP target. The task has
# called it both "tcp_target" (older checkpoints) and "target" (spirob task
# restructure); both are the same internal value, built from the target x/z.
_TARGET_TERMS = ("tcp_target", "target")

# Where each actor term's value comes from:
#   None            -> filled internally by the bridge (target / last action)
#   "<signal name>" -> pulled from whichever connected source provides it
# A term absent here is not realizable on hardware (needs a sensor the rig
# lacks) and makes validation fail loudly.
_TERM_SIGNAL: dict[str, str | None] = {
  "tcp_target": None,
  "target": None,
  "last_action": None,
  "tendon_len": "tendon_len",
  "tendon_vel": "tendon_vel",
  "joint_pos": "joint_pos",
  "joint_vel": "joint_vel",
  "segment_pitch": "segment_pitch",
  # tcp_pos is not read from a live source but computed from the joint angles by
  # forward kinematics. Only the online-training env supplies it (for the
  # critic); the inference bridge rejects it (see _INFERENCE_ONLY below).
  "tcp_pos": "tcp_pos",
}

# Terms that no live source produces -- only the training env computes them
# (forward kinematics on the measured joint angles). The inference bridge treats
# them as not realizable so an oracle-level actor still fails with a clear
# message instead of a confusing "missing signal".
_INFERENCE_ONLY: dict[str, str] = {
  "tcp_pos": "the privileged TCP pose; only the online-training env computes it "
  "via forward kinematics, the inference bridge cannot",
}


@dataclass(frozen=True)
class TermSpec:
  name: str
  base_dim: int  # dimension of one frame, before history stacking
  history_length: int


@dataclass(frozen=True)
class ObsLayout:
  """Layout of one observation group (actor or critic): its terms in order."""

  group: str
  terms: tuple[TermSpec, ...]
  total_dim: int
  sim_joint_names: tuple[str, ...]

  @property
  def action_dim(self) -> int:
    for term in self.terms:
      if term.name == "last_action":
        return term.base_dim
    raise KeyError(f"{self.group} layout has no 'last_action' term to infer action dim from")


# Backwards-compatible alias: the inference bridge imports ``ActorLayout``.
ActorLayout = ObsLayout


def _read_group_layout(om, group: str, sim_joint_names: tuple[str, ...]) -> ObsLayout:
  names = om._group_obs_term_names[group]
  flat_dims = om._group_obs_term_dim[group]
  term_cfgs = om._group_obs_term_cfgs[group]
  terms: list[TermSpec] = []
  for name, flat, term_cfg in zip(names, flat_dims, term_cfgs):
    history = max(int(term_cfg.history_length), 1)
    flat_dim = int(flat[0]) if isinstance(flat, tuple) else int(flat)
    terms.append(TermSpec(name=name, base_dim=flat_dim // history, history_length=history))
  total = sum(int(f[0]) if isinstance(f, tuple) else int(f) for f in flat_dims)
  return ObsLayout(
    group=group, terms=tuple(terms), total_dim=total, sim_joint_names=sim_joint_names
  )


def derive_layouts(task_id: str, device: str, groups: tuple[str, ...] = ("actor",)) -> dict[str, ObsLayout]:
  """Build the task env once and read the requested observation groups' layouts.

  This is the same env construction ``spirob_rl.infer.load_policy`` does,
  paid once at startup (a few seconds) to keep this module independent of the
  policy-loading path. Nothing here runs in the control loop.
  """
  # Local import: pulls in the full sim/task stack, which the loop never needs.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import spirob_rl.tasks  # noqa: F401  (registers the tasks)

  cfg = load_env_cfg(task_id, play=True)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    _, joint_names = env.scene["spirob"].find_joints(("j_.*",))
    joint_names = tuple(joint_names)
    return {g: _read_group_layout(env.observation_manager, g, joint_names) for g in groups}
  finally:
    env.close()


def derive_actor_layout(task_id: str, device: str) -> ObsLayout:
  """Read just the actor layout (the inference bridge's entry point)."""
  return derive_layouts(task_id, device, groups=("actor",))["actor"]


def required_signals(layout: ActorLayout) -> set[str]:
  """Signal names this policy needs from hardware sources (excludes internals)."""
  signals: set[str] = set()
  for term in layout.terms:
    signal = _TERM_SIGNAL.get(term.name)
    if signal is not None:
      signals.add(signal)
  return signals


def validate(layout: ActorLayout, sources) -> None:
  """Fail (before any motion) unless the connected sources satisfy every term.

  ``sources`` is any iterable of hardware sources with ``.name`` and
  ``.provides``.
  """
  available: dict[str, str] = {}
  for source in sources:
    for signal in source.provides:
      available.setdefault(signal, source.name)

  for term in layout.terms:
    if term.name in _INFERENCE_ONLY:
      raise SystemExit(
        f"This policy's actor needs {term.name!r} ({_INFERENCE_ONLY[term.name]}). "
        "Pick a policy whose sensor level only uses live-measurable terms "
        "(force, tendon, joints, imu)."
      )
    if term.name not in _TERM_SIGNAL:
      raise SystemExit(
        f"This policy needs the observation term {term.name!r}, which no hardware "
        "source can produce. Pick a policy whose sensor level only uses realizable "
        "terms (force, tendon, joints, imu)."
      )
    signal = _TERM_SIGNAL[term.name]
    if signal is None:
      continue
    if signal not in available:
      # Name the source that would provide it rather than a generic "connect
      # something": every missing signal today comes from one of two boards.
      from .sources import JointSensor, MotorRig

      if signal in JointSensor.provides:
        hint = "the accelerometer board -- pass --joint-port /dev/ttyUSB1"
      elif signal in MotorRig.provides:
        hint = "the motor MCU -- pass --port /dev/ttyUSB0"
      else:
        hint = "no source in sources.py provides it"
      raise SystemExit(
        f"This policy needs the observation term {term.name!r} (signal "
        f"{signal!r}), but no connected source provides it. It comes from {hint}."
      )


class ObservationAssembler:
  """Rebuilds the actor observation vector each tick from live signals.

  History handling mirrors mjlab's ``CircularBuffer``: the first frame is
  backfilled across every history slot (so the policy never sees zero-padded
  history right after start), then a ``deque(maxlen=...)`` gives FIFO
  oldest->newest eviction. Terms are concatenated in the layout's order, each
  with its own history flattened -- exactly what the observation manager
  produces in sim.
  """

  def __init__(self, layout: ActorLayout, device: str) -> None:
    self.layout = layout
    self.device = device
    self._history: dict[str, deque] = {
      term.name: deque(maxlen=term.history_length) for term in layout.terms
    }
    self._dims_checked = False

  def _push(self, name: str, value: list[float]) -> None:
    dq = self._history[name]
    if len(dq) == 0:
      dq.extend([value] * dq.maxlen)
    else:
      dq.append(value)

  def assemble(
    self,
    target_xz: list[float],
    last_raw_action: list[float],
    signals: dict[str, list[float]],
  ) -> torch.Tensor:
    flat: list[float] = []
    for term in self.layout.terms:
      signal = _TERM_SIGNAL[term.name]
      if term.name in _TARGET_TERMS:
        value = [target_xz[0], 0.0, target_xz[1]]
      elif term.name == "last_action":
        value = list(last_raw_action)
      else:
        assert signal is not None
        value = list(signals[signal])

      if len(value) != term.base_dim:
        raise SystemExit(
          f"Signal for term {term.name!r} has dim {len(value)}, but the policy "
          f"expects {term.base_dim}. A hardware source and the sim disagree on "
          f"this term's size (e.g. joint count)."
        )
      self._push(term.name, value)

    for term in self.layout.terms:
      for frame in self._history[term.name]:
        flat.extend(frame)
    return torch.tensor([flat], dtype=torch.float32, device=self.device)
