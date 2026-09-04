"""Sweep a trained reach policy over the whole workspace and measure where it lands.

The reach task draws one random TCP target per episode, so training and play
only ever show the workspace *sampled*: a good run tells you the policy works
somewhere, not where in the shell it works well and where it does not. This
module turns that around and asks the question systematically -- lay a grid over
the trained target distribution, drive the policy to every grid point, and
record how far the TCP still is from the target once it has settled.

Two things make the resulting map trustworthy:

* **Every grid point is approached from the same set of start postures.** The
  reset event draws 13 independent joint offsets, and a tentacle that happens to
  start leaning toward its target has an easier time than one starting away from
  it. Sampling starts freshly per target would fold that luck into the map and
  show it as a property of the *target*. So the start postures are drawn once
  from a seeded RNG (``--repeats`` of them) and every target sees exactly that
  same set -- a paired design, in which the spread across repeats becomes its own
  readable quantity (``spread`` in the figure) rather than noise on the mean.
* **Nothing else varies.** The play config is used as-is (no domain
  randomization, no observation noise), the command term's resampling is
  disabled and its targets are overwritten with the grid, so each environment
  holds one target for the whole rollout.

The grid lives in the shell's own polar coordinates -- angle from +z, and signed
deviation from the shell radius ``r(angle) = shell_radius - shell_curvature *
angle**2`` -- read off the task's own ``TcpPositionCommandCfg`` rather than
copied, so the sweep always covers exactly the distribution the policy was
trained on. ``--angle-scale`` / ``--band-scale`` above 1.0 deliberately step
outside it to probe extrapolation.

Output is one ``.npz`` per sweep in ``build/rl/workspace/`` holding the full
distance-over-time trace for every (target, start posture) pair, which
:mod:`spirob_rl.rig.workspace_figure` renders and any later question can be re-asked
from without touching the simulator again.

Usage::

    uv run python -m spirob_rl.rig.workspace_sweep
    uv run python -m spirob_rl.rig.workspace_sweep --task RlExplor-Spirob-Tcp-Reach-Joints \\
        --checkpoint build/rl/logs/rl_explor_spirob_tcp_joints/<run>/model_499.pt \\
        --angles 25 --radii 7 --repeats 8 --settle 15 --window 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from spirob.paths import build_dir

from spirob_rl.cli import LOG_ROOT

BUILD_DIR = build_dir("rl", "workspace")

DEFAULT_TASK = "RlExplor-Spirob-Tcp-Reach-Joints"
"""Sensor level "joints": TCP target plus the 13 joint angles and velocities.

The level the thesis' workspace map is about -- realizable on the rig via the
accelerometer board, unlike ``oracle``, and the one whose trained policy places
the tip anywhere in the shell.
"""

COMMAND_NAME = "tcp_target"
RESET_TERM = "reset_joints"


# -- grid ---------------------------------------------------------------------


@dataclass(frozen=True)
class ShellGeometry:
  """The workspace shell the policy was trained on, in its own coordinates."""

  angle_limit: float
  shell_radius: float
  shell_curvature: float
  shell_band: float

  def radius_at(self, angle: np.ndarray | float) -> np.ndarray:
    return self.shell_radius - self.shell_curvature * np.asarray(angle) ** 2

  def to_xz(self, angle: np.ndarray, deviation: np.ndarray) -> np.ndarray:
    """Shell polar coordinates to (x, z). Broadcasts; last axis is (x, z)."""
    radius = self.radius_at(angle) + deviation
    return np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=-1)


def geometry_from_cfg(cmd_cfg) -> ShellGeometry:
  """Read the shell off the task's own command config -- never a copy of it."""
  lo, hi = cmd_cfg.angle_range
  return ShellGeometry(
    angle_limit=float(max(abs(lo), abs(hi))),
    shell_radius=float(cmd_cfg.shell_radius),
    shell_curvature=float(cmd_cfg.shell_curvature),
    shell_band=float(cmd_cfg.shell_band),
  )


def make_grid(
  geometry: ShellGeometry,
  n_angle: int,
  n_radius: int,
  angle_scale: float = 1.0,
  band_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Grid centres in shell coordinates: ``(angles [n_angle], deviations [n_radius])``.

  ``n_radius == 1`` collapses the band to its centreline, which is the sensible
  degenerate case rather than an error: the shell is thin, and a map of the
  centreline alone is a legitimate (cheaper) question.
  """
  limit = geometry.angle_limit * angle_scale
  band = geometry.shell_band * band_scale
  angles = np.linspace(-limit, limit, n_angle) if n_angle > 1 else np.zeros(1)
  deviations = np.linspace(-band, band, n_radius) if n_radius > 1 else np.zeros(1)
  return angles, deviations


def make_start_poses(
  n_repeats: int, n_joints: int, position_range: tuple[float, float], seed: int
) -> np.ndarray:
  """``[n_repeats, n_joints]`` joint offsets, the same set for every target.

  Drawn from the same uniform range the reset event uses, so the postures are
  ordinary draws from the training reset distribution -- only fixed, not fresh
  per target. See the module docstring for why that matters.
  """
  rng = np.random.default_rng(seed)
  lo, hi = position_range
  return rng.uniform(lo, hi, size=(n_repeats, n_joints)).astype(np.float32)


# -- result -------------------------------------------------------------------


@dataclass
class SweepResult:
  """Everything a figure or a later question could want from one sweep."""

  angles: np.ndarray
  """Grid angles from +z [rad]. ``[n_angle]``"""
  deviations: np.ndarray
  """Grid deviations from the shell radius [m]. ``[n_radius]``"""
  distance: np.ndarray
  """TCP-target distance over time [m]. ``[n_angle, n_radius, n_repeats, n_steps]``"""
  tcp_final: np.ndarray
  """Where the TCP ended up, (x, z) [m]. ``[n_angle, n_radius, n_repeats, 2]``"""
  start_poses: np.ndarray
  """Joint offsets of each repeat [rad]. ``[n_repeats, n_joints]``"""
  dt: float
  """Seconds per recorded step (one policy step)."""
  window: float
  """Length of the averaging window at the end of the rollout [s]."""
  success_threshold: float
  """Distance below which a target counts as reached [m]."""
  geometry: ShellGeometry
  meta: dict

  # -- derived quantities, all over the (target, repeat) grid ------------------

  @property
  def n_window_steps(self) -> int:
    return max(1, round(self.window / self.dt))

  @property
  def final_error(self) -> np.ndarray:
    """Mean distance over the averaging window. ``[n_angle, n_radius, n_repeats]``

    A window rather than the last sample: a policy that orbits its target would
    otherwise be scored on where in the orbit the rollout happened to stop.
    """
    return self.distance[..., -self.n_window_steps :].mean(axis=-1)

  @property
  def best_error(self) -> np.ndarray:
    """Closest approach at any time. ``[n_angle, n_radius, n_repeats]``

    Separates "never got there" from "got there and drifted off again".
    """
    return self.distance.min(axis=-1)

  @property
  def mean_error(self) -> np.ndarray:
    """Final error averaged over start postures. ``[n_angle, n_radius]``"""
    return self.final_error.mean(axis=-1)

  @property
  def spread(self) -> np.ndarray:
    """Std of the final error across start postures. ``[n_angle, n_radius]``

    How much the start posture decides the outcome at this target -- the paired
    design is what makes this readable as a property of the target.
    """
    return self.final_error.std(axis=-1)

  @property
  def worst_error(self) -> np.ndarray:
    return self.final_error.max(axis=-1)

  @property
  def success_rate(self) -> np.ndarray:
    """Fraction of start postures whose final error is below the threshold."""
    return (self.final_error < self.success_threshold).mean(axis=-1)

  @property
  def settle_time(self) -> np.ndarray:
    """First time the distance falls below the threshold [s]; NaN if never.

    ``[n_angle, n_radius, n_repeats]``
    """
    below = self.distance < self.success_threshold
    first = np.argmax(below, axis=-1).astype(np.float64)
    first[~below.any(axis=-1)] = np.nan
    return first * self.dt

  @property
  def targets_xz(self) -> np.ndarray:
    """Commanded target positions. ``[n_angle, n_radius, 2]``"""
    return self.geometry.to_xz(self.angles[:, None], self.deviations[None, :])

  # -- storage -----------------------------------------------------------------

  def save(self, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      path,
      angles=self.angles,
      deviations=self.deviations,
      distance=self.distance.astype(np.float32),
      tcp_final=self.tcp_final.astype(np.float32),
      start_poses=self.start_poses.astype(np.float32),
      scalars=np.array(
        json.dumps({
          "dt": self.dt,
          "window": self.window,
          "success_threshold": self.success_threshold,
          "geometry": asdict(self.geometry),
          "meta": self.meta,
        })
      ),
    )
    return path

  @classmethod
  def load(cls, path: Path) -> "SweepResult":
    with np.load(Path(path), allow_pickle=False) as data:
      scalars = json.loads(str(data["scalars"]))
      return cls(
        angles=data["angles"],
        deviations=data["deviations"],
        distance=data["distance"],
        tcp_final=data["tcp_final"],
        start_poses=data["start_poses"],
        dt=float(scalars["dt"]),
        window=float(scalars["window"]),
        success_threshold=float(scalars["success_threshold"]),
        geometry=ShellGeometry(**scalars["geometry"]),
        meta=scalars["meta"],
      )


def summary_text(result: SweepResult) -> str:
  """A few lines that say what the map says, for the console and the log."""
  error_mm = result.final_error * 1000.0
  mean_mm = result.mean_error * 1000.0
  lines = [
    f"targets            {len(result.angles)} angles x {len(result.deviations)} radii"
    f" x {result.start_poses.shape[0]} start postures"
    f" = {error_mm.size} rollouts",
    f"final error        median {np.median(error_mm):6.1f} mm"
    f"   mean {error_mm.mean():6.1f} mm"
    f"   p90 {np.percentile(error_mm, 90):6.1f} mm"
    f"   max {error_mm.max():6.1f} mm",
    f"success (<{result.success_threshold * 1000:.0f} mm)  "
    f"{(result.final_error < result.success_threshold).mean() * 100:5.1f} % of rollouts,"
    f" {(result.success_rate == 1.0).mean() * 100:5.1f} % of targets from every start",
    f"start posture      spread across repeats: median"
    f" {np.median(result.spread) * 1000:5.1f} mm, max {result.spread.max() * 1000:5.1f} mm",
  ]
  best_idx = np.unravel_index(np.argmin(mean_mm), mean_mm.shape)
  worst_idx = np.unravel_index(np.argmax(mean_mm), mean_mm.shape)
  for label, idx in (("best   target", best_idx), ("worst  target", worst_idx)):
    angle = result.angles[idx[0]]
    deviation = result.deviations[idx[1]]
    x, z = result.geometry.to_xz(np.array(angle), np.array(deviation))
    lines.append(
      f"{label}      angle {angle:+.2f} rad, deviation {deviation * 1000:+5.0f} mm"
      f"  (x={x:+.3f}, z={z:.3f}) -> {mean_mm[idx]:6.1f} mm"
    )
  return "\n".join(lines)


# -- the sweep itself ---------------------------------------------------------


def resolve_checkpoint(task_id: str, checkpoint: Optional[str], log_root: Path) -> Path:
  """Explicit path, or the highest ``model_<n>.pt`` of the newest run of the task.

  Deliberately not ``mjlab.utils.os.get_checkpoint_path``: its default regexes
  match every file in the run directory, so the newest "checkpoint" it finds can
  be ``params/``.
  """
  if checkpoint:
    path = Path(checkpoint)
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path

  from mjlab.tasks.registry import load_rl_cfg

  experiment = load_rl_cfg(task_id).experiment_name
  experiment_dir = log_root / experiment
  if not experiment_dir.is_dir():
    raise FileNotFoundError(
      f"No logs for task {task_id} in {experiment_dir}. Pass --checkpoint."
    )
  runs = sorted(
    (d for d in experiment_dir.iterdir() if d.is_dir() and d.name != "wandb_checkpoints"),
    key=lambda d: d.stat().st_mtime,
  )
  for run in reversed(runs):
    models = [
      (int(m.group(1)), f)
      for f in run.iterdir()
      if (m := re.fullmatch(r"model_(\d+)\.pt", f.name))
    ]
    if models:
      return max(models)[1]
  raise FileNotFoundError(f"No model_<n>.pt checkpoint below {experiment_dir}.")


def _install_grid_targets(env, targets_holder: list) -> None:
  """Make the reach command serve the sweep's grid instead of random draws.

  Overriding ``_resample_command`` rather than writing ``target_pos_w`` after
  the reset is what keeps the observation history clean: the history buffer is
  backfilled during ``reset()`` with whatever the command said *then*, so a
  target written afterwards would leave the policy's first frames describing a
  target that was never commanded.
  """
  import torch

  term = env.command_manager.get_term(COMMAND_NAME)

  def _resample(env_ids: "torch.Tensor") -> None:
    targets = targets_holder[0]  # [num_envs, 3], already in world coordinates
    term.target_pos_w[env_ids, 0] = targets[env_ids]

  term._resample_command = _resample  # type: ignore[method-assign]
  # Belt and braces: the grid is also immune to the resampling timer firing.
  term.cfg.resampling_time_range = (1.0e9, 1.0e9)


def _install_fixed_start_poses(env, offsets_holder: list):
  """Replace the random reset offsets with the sweep's fixed posture table.

  Returns the number of joints the table has to have.
  """
  import torch

  term_cfg = env.event_manager.get_term_cfg(RESET_TERM)
  asset_cfg = term_cfg.params["asset_cfg"]
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  def _reset(env, env_ids, **_) -> None:
    if not isinstance(env_ids, torch.Tensor):
      env_ids = torch.arange(env.num_envs, device=env.device)
    offsets = offsets_holder[0]  # [num_envs, n_joints]
    position = asset.data.default_joint_pos[env_ids][:, joint_ids] + offsets[env_ids]
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_state_to_sim(
      position, torch.zeros_like(position), env_ids=env_ids, joint_ids=joint_ids
    )

  term_cfg.func = _reset
  term_cfg.params = {}
  return asset.data.default_joint_pos[:, joint_ids].shape[1]


def run_sweep(
  task_id: str = DEFAULT_TASK,
  checkpoint: Optional[str] = None,
  n_angle: int = 25,
  n_radius: int = 7,
  n_repeats: int = 8,
  settle: float = 15.0,
  window: float = 3.0,
  angle_scale: float = 1.0,
  band_scale: float = 1.0,
  max_envs: int = 2048,
  seed: int = 0,
  device: Optional[str] = None,
  log_root: Path = LOG_ROOT,
) -> SweepResult:
  """Drive the policy to every grid point from every start posture."""
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
  )
  from mjlab.utils.torch import configure_torch_backends

  import spirob_rl.tasks as _tasks  # noqa: F401  (registers the tasks)

  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint_path = resolve_checkpoint(task_id, checkpoint, log_root)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  geometry = geometry_from_cfg(env_cfg.commands[COMMAND_NAME])
  success_threshold = float(env_cfg.commands[COMMAND_NAME].success_threshold)
  position_range = tuple(env_cfg.events[RESET_TERM].params["position_range"])

  angles, deviations = make_grid(
    geometry, n_angle, n_radius, angle_scale=angle_scale, band_scale=band_scale
  )
  n_angle, n_radius = len(angles), len(deviations)
  n_targets = n_angle * n_radius
  n_rollouts = n_targets * n_repeats

  # One environment per (target, start posture) pair, capped at --max-envs and
  # run in as many batches as that takes. Envs of the last batch beyond the
  # remaining rollouts still simulate; their results are dropped.
  num_envs = min(max_envs, n_rollouts)
  env_cfg.scene.num_envs = num_envs

  print(f"[sweep] task        {task_id}")
  print(f"[sweep] checkpoint  {checkpoint_path}")
  print(
    f"[sweep] grid        {n_angle} angles x {n_radius} radii x {n_repeats} starts"
    f" = {n_rollouts} rollouts in {int(np.ceil(n_rollouts / num_envs))} batch(es)"
    f" of {num_envs} envs"
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  targets_holder: list = [torch.zeros(num_envs, 3, device=device)]
  offsets_holder: list = [None]
  _install_grid_targets(env, targets_holder)
  n_joints = _install_fixed_start_poses(env, offsets_holder)

  start_poses = make_start_poses(n_repeats, n_joints, position_range, seed)
  start_poses_t = torch.as_tensor(start_poses, device=device)

  dt = float(env.step_dt)
  n_steps = max(1, int(round(settle / dt)))

  # Flat rollout index -> (angle, radius, repeat), repeat fastest. Keeping the
  # repeats of one target inside one batch is not required, but it means a
  # partial run still covers whole targets.
  flat_angle, flat_radius, flat_repeat = (
    idx.reshape(-1)
    for idx in np.meshgrid(
      np.arange(n_angle), np.arange(n_radius), np.arange(n_repeats), indexing="ij"
    )
  )
  target_xz = geometry.to_xz(angles[flat_angle], deviations[flat_radius])  # [n_rollouts, 2]

  distance = np.zeros((n_rollouts, n_steps), dtype=np.float32)
  tcp_final = np.zeros((n_rollouts, 2), dtype=np.float32)

  command_term = env.command_manager.get_term(COMMAND_NAME)
  started = time.time()
  for batch_start in range(0, n_rollouts, num_envs):
    batch = np.arange(batch_start, min(batch_start + num_envs, n_rollouts))
    # Pad a short final batch by repeating its first entry; those envs are simulated
    # but never read.
    padded = np.concatenate([batch, np.full(num_envs - len(batch), batch[0])])

    xz = torch.as_tensor(target_xz[padded], dtype=torch.float32, device=device)
    targets = torch.zeros(num_envs, 3, device=device)
    targets[:, 0] = xz[:, 0]
    targets[:, 2] = xz[:, 1]
    targets_holder[0] = targets + env.scene.env_origins
    offsets_holder[0] = start_poses_t[torch.as_tensor(flat_repeat[padded], device=device)]

    obs, _ = wrapped.reset()
    for step in range(n_steps):
      with torch.inference_mode():
        actions = policy(obs)
      obs, _, _, _ = wrapped.step(actions)
      error = torch.norm(
        command_term.target_pos_w[:, 0] - command_term.tcp_pos_w, dim=-1
      )
      distance[batch, step] = error[: len(batch)].cpu().numpy()

    tcp_local = command_term.tcp_pos_w - env.scene.env_origins
    tcp_final[batch] = tcp_local[: len(batch)][:, [0, 2]].cpu().numpy()

    done = min(batch_start + num_envs, n_rollouts)
    elapsed = time.time() - started
    print(
      f"[sweep] {done}/{n_rollouts} rollouts, {elapsed:5.1f} s"
      f" (eta {elapsed * (n_rollouts / done - 1):5.1f} s)"
    )

  env.close()

  shape = (n_angle, n_radius, n_repeats)
  result = SweepResult(
    angles=angles,
    deviations=deviations,
    distance=distance.reshape(*shape, n_steps),
    tcp_final=tcp_final.reshape(*shape, 2),
    start_poses=start_poses,
    dt=dt,
    window=window,
    success_threshold=success_threshold,
    geometry=geometry,
    meta={
      "task": task_id,
      "checkpoint": str(checkpoint_path),
      "settle_s": settle,
      "angle_scale": angle_scale,
      "band_scale": band_scale,
      "seed": seed,
      "device": device,
      "reset_position_range": list(position_range),
      "recorded": datetime.now().isoformat(timespec="seconds"),
    },
  )
  return result


def default_out_base() -> Path:
  return BUILD_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}"


def main(argv: Optional[list[str]] = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--checkpoint", default=None,
                      help="checkpoint file; default is the newest run's last model")
  parser.add_argument("--angles", type=int, default=25, help="grid points along the shell")
  parser.add_argument("--radii", type=int, default=7, help="grid points across the band")
  parser.add_argument("--repeats", type=int, default=8,
                      help="start postures per target (the same set for all targets)")
  parser.add_argument("--settle", type=float, default=15.0,
                      help="rollout length per target [s]")
  parser.add_argument("--window", type=float, default=3.0,
                      help="averaging window at the end of the rollout [s]")
  parser.add_argument("--angle-scale", type=float, default=1.0,
                      help=">1 sweeps beyond the trained angle limit")
  parser.add_argument("--band-scale", type=float, default=1.0,
                      help=">1 sweeps beyond the trained shell band")
  parser.add_argument("--max-envs", type=int, default=2048, help="parallel envs per batch")
  parser.add_argument("--seed", type=int, default=0, help="seed of the start-posture table")
  parser.add_argument("--device", default=None)
  parser.add_argument("--out", default=None, help="output .npz (default: timestamped)")
  parser.add_argument("--figure", action="store_true",
                      help="render the figures right after the sweep")
  args = parser.parse_args(argv)

  result = run_sweep(
    task_id=args.task,
    checkpoint=args.checkpoint,
    n_angle=args.angles,
    n_radius=args.radii,
    n_repeats=args.repeats,
    settle=args.settle,
    window=args.window,
    angle_scale=args.angle_scale,
    band_scale=args.band_scale,
    max_envs=args.max_envs,
    seed=args.seed,
    device=args.device,
  )

  out = Path(args.out) if args.out else default_out_base().with_suffix(".npz")
  path = result.save(out)
  print(f"\n{summary_text(result)}\n")
  print(f"[sweep] {path}")

  if args.figure:
    from . import workspace_figure

    workspace_figure.save_figures(result, out_base=path.with_suffix(""))


if __name__ == "__main__":
  main()
