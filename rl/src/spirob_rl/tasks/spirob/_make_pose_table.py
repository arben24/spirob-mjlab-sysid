"""Regenerate ``holdable_poses.npz``, the pose table used by the shape task.

Run after changing the spirob model or its dynamic parameters::

    uv run python -m spirob_rl.tasks.spirob._make_pose_table

The tentacle has 13 joints but only 2 tendon forces, so the poses it can
statically hold form at most a 2-manifold in joint space -- the image of the
control square under the quasi-static map. Analytic guesses at that manifold fit
badly (a linear curvature+taper family leaves ~0.2 rad residual per joint,
against a 0.51 rad joint limit), so it is measured: sweep a grid over the two
tendon commands, let each settle, record the pose.

The base joint j_12 is almost undamped in the XML (1e-4), so some poses keep
ringing rather than coming to rest. The recorded value is the mean over the last
second -- the pose the tentacle oscillates *about* -- and the amplitude is
reported so the residual is visible rather than hidden.

The leading underscore keeps ``spirob_rl.tasks``' auto-import walker from
importing (and thereby running) this module at task-registration time.
"""

from __future__ import annotations

import numpy as np
import torch

import spirob_rl.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from .mdp.constants import HOLDABLE_POSES_NPZ

GRID = 24  # GRID^2 environments, one per tendon-command pair
SETTLE_STEPS = 400  # 8 s at 50 Hz
AVG_STEPS = 50  # average the pose over the last second


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg("RlExplor-Spirob-Tcp-Reach", play=True)
  cfg.scene.num_envs = GRID * GRID
  cfg.episode_length_s = 1e10
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  robot = env.scene["spirob"]

  # Grid over the two tendon commands in action space [-1, 1].
  axis = torch.linspace(-1.0, 1.0, GRID, device=device)
  a0, a1 = torch.meshgrid(axis, axis, indexing="ij")
  actions = torch.stack([a0.flatten(), a1.flatten()], dim=-1)

  for _ in range(SETTLE_STEPS - AVG_STEPS):
    env.step(actions)
  window = []
  for _ in range(AVG_STEPS):
    env.step(actions)
    window.append(robot.data.joint_pos.clone())

  stack = torch.stack(window)  # [T, N, num_joints]
  q_mean = stack.mean(0)
  oscillation = stack.max(0).values - stack.min(0).values

  print(f"poses: {q_mean.shape[0]}  joints: {q_mean.shape[1]}")
  print("residual oscillation (peak-to-peak over the last second):")
  print(
    f"  mean {oscillation.mean():.4f} rad, "
    f"q95 {torch.quantile(oscillation.flatten(), 0.95):.4f}, "
    f"max {oscillation.max():.4f}"
  )
  print(f"joint range: [{q_mean.min():+.4f}, {q_mean.max():+.4f}]")

  table = q_mean.reshape(GRID, GRID, -1).cpu().numpy().astype(np.float32)
  np.savez_compressed(
    HOLDABLE_POSES_NPZ,
    joint_pos=table,
    oscillation=oscillation.reshape(GRID, GRID, -1).cpu().numpy().astype(np.float32),
  )
  print(f"wrote {HOLDABLE_POSES_NPZ}  shape={table.shape}")
  env.close()


if __name__ == "__main__":
  main()
