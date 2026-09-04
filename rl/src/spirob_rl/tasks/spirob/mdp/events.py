"""Event terms for the spirob task family."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class grid_layout:
  """Spread parallel environments out in a square grid, spaced by
  ``SceneCfg.env_spacing`` (in meters), so multiple envs don't visually
  overlap when watched together (e.g. DR-play with several tentacles).

  mjlab only auto-populates ``scene.env_origins`` via a ``TerrainEntityCfg``,
  and even that would only ever move the *terrain*: a fixed-base entity like
  the spirob (no free joint on its base) has its body position baked into the
  compiled model at build time, identical across every parallel world, so
  terrain alone would not separate the tentacles. This term does both parts
  needed for a real grid: it writes the "base" body's position per world
  directly into the (now per-world-expanded) ``body_pos`` model field, and
  mirrors the same offsets into ``scene.env_origins`` so our own
  observations/commands -- which already treat positions as relative to
  ``env.scene.env_origins`` -- stay numerically identical (a constant per-env
  offset cancels out in every difference we compute, e.g. target - tcp).
  Runs once at ``mode="startup"``, before the first reset.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    if env.num_envs <= 1:
      return

    body_id = int(cfg.params["asset_cfg"].body_ids[0])
    spacing = env.scene.env_spacing

    num_cols = int(math.ceil(math.sqrt(env.num_envs)))
    num_rows = int(math.ceil(env.num_envs / num_cols))
    ii, jj = torch.meshgrid(
      torch.arange(num_rows, device=env.device),
      torch.arange(num_cols, device=env.device),
      indexing="ij",
    )
    origins = torch.zeros(env.num_envs, 3, device=env.device)
    origins[:, 0] = -(ii.flatten()[: env.num_envs] - (num_rows - 1) / 2) * spacing
    origins[:, 1] = (jj.flatten()[: env.num_envs] - (num_cols - 1) / 2) * spacing

    env.sim.expand_model_fields(("body_pos",))
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.sim.model.body_pos[env_ids, body_id] += origins

    if env.scene._terrain is None:
      env.scene._default_env_origins = origins

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids, asset_cfg: SceneEntityCfg
  ) -> None:
    del env, env_ids, asset_cfg  # Everything happens once in __init__.
