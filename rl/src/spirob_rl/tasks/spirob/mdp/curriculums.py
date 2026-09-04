"""Curriculum terms for the spirob task family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class dr_range_curriculum:
  """Widen domain-randomization ranges over the course of training.

  Each managed event term's ``ranges`` param is linearly interpolated from a
  no-op width at ``start_step`` to the term's target range at ``end_step``,
  keyed on ``env.common_step_counter`` (which counts policy steps, so the
  horizon is ``max_iterations * num_steps_per_env``). Training therefore starts
  at the nominal XML values -- zero spread -- and the random draw widens as it
  proceeds.

  ``targets`` maps event-term name -> final ``(lo, hi)``. ``start_ranges`` is
  the no-op width the ramp starts from: ``(1.0, 1.0)`` for ``operation="scale"``
  (multiplying by 1 leaves the value unchanged). Terms listed in ``targets``
  but not currently active (commented out) are skipped, so toggling a DR term
  on or off never breaks this. Only terms that take a ``ranges`` param with
  ``scale`` semantics belong here -- not ``pseudo_inertia`` (``alpha_range``)
  or ``add``-based play terms, which start from a different no-op width.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._targets: dict[str, tuple[float, float]] = cfg.params["targets"]
    self._start_step: int = cfg.params["start_step"]
    self._end_step: int = cfg.params["end_step"]
    self._start_ranges: tuple[float, float] = cfg.params.get(
      "start_ranges", (1.0, 1.0)
    )
    active = set(env.event_manager.active_terms.get("reset", []))
    self._term_cfgs = {
      name: env.event_manager.get_term_cfg(name)
      for name in self._targets
      if name in active
    }

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    targets: dict[str, tuple[float, float]],
    start_step: int,
    end_step: int,
    start_ranges: tuple[float, float] = (1.0, 1.0),
  ) -> dict[str, torch.Tensor]:
    del env_ids, targets, start_step, end_step, start_ranges
    span = max(self._end_step - self._start_step, 1)
    alpha = (env.common_step_counter - self._start_step) / span
    alpha = float(min(max(alpha, 0.0), 1.0))
    s_lo, s_hi = self._start_ranges
    for name, term_cfg in self._term_cfgs.items():
      final_lo, final_hi = self._targets[name]
      term_cfg.params["ranges"] = (
        s_lo + alpha * (final_lo - s_lo),
        s_hi + alpha * (final_hi - s_hi),
      )
    return {"dr_curriculum/alpha": torch.tensor(alpha)}
