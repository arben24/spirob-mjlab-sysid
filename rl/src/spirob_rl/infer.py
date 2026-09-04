"""Load a trained policy checkpoint and run it on external observations.

This script reuses the existing mjlab / rsl_rl runner path to construct and
load the policy, then detaches the policy from the simulation environment so it
can be fed with your own observation tensors on real hardware or from files.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from . import tasks as _tasks  # noqa: F401
from .cli import LOG_ROOT


@dataclass(frozen=True)
class InferConfig:
  """Configuration for policy inference."""

  checkpoint_file: str | None = None
  """Explicit checkpoint path. If omitted, the latest checkpoint in logs is used."""

  log_root: str = str(LOG_ROOT)
  """Where to look for runs when no explicit checkpoint is given. Defaults to
  the same ``build/rl/logs`` the entrypoints train into (see ``cli.py``)."""

  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""

  obs_file: str | None = None
  """Optional observation input file (.npy, .npz, .json, .jsonl, or .pt)."""

  output_file: str | None = None
  """Optional file to save the computed actions to."""

  stream_stdin: bool = False
  """Read one observation packet per line from stdin and write actions to stdout."""


def _load_payload(path: Path) -> Any:
  suffix = path.suffix.lower()
  if suffix == ".npy":
    return np.load(path, allow_pickle=False)
  if suffix == ".npz":
    with np.load(path, allow_pickle=False) as data:
      if len(data.files) == 1:
        return data[data.files[0]]
      return {name: data[name] for name in data.files}
  if suffix in {".pt", ".pth"}:
    return torch.load(path, map_location="cpu", weights_only=False)
  if suffix in {".jsonl", ".ndjson"}:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if line:
          rows.append(json.loads(line))
    return rows
  if suffix == ".json" or suffix == "":
    with path.open("r", encoding="utf-8") as handle:
      return json.load(handle)

  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def _as_float_tensor(value: Any, device: str) -> torch.Tensor:
  tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
  if tensor.ndim == 0:
    return tensor.reshape(1, 1)
  if tensor.ndim == 1:
    return tensor.unsqueeze(0)
  return tensor


def _payload_to_inputs(payload: Any, policy, device: str) -> dict[str, torch.Tensor]:
  expected_groups = list(getattr(policy, "obs_groups", ["actor"]))

  if isinstance(payload, dict):
    inputs: dict[str, torch.Tensor] = {}
    if set(expected_groups).issubset(payload.keys()):
      for group_name in expected_groups:
        inputs[group_name] = _as_float_tensor(payload[group_name], device)
      return inputs

    if len(payload) == 1:
      only_value = next(iter(payload.values()))
      inputs[expected_groups[0]] = _as_float_tensor(only_value, device)
      return inputs

    if "obs" in payload and len(expected_groups) == 1:
      inputs[expected_groups[0]] = _as_float_tensor(payload["obs"], device)
      return inputs

    missing = [name for name in expected_groups if name not in payload]
    raise ValueError(
      "Observation payload must either provide all policy observation groups "
      f"({expected_groups}) or a single flat vector. Missing groups: {missing}"
    )

  if isinstance(payload, list):
    if payload and isinstance(payload[0], dict):
      if len(expected_groups) != 1:
        raise ValueError(
          "JSON lines batches are only supported for single-group actor policies."
        )
      stacked = [
        _as_float_tensor(row.get("obs", next(iter(row.values()))), device)
        for row in payload
      ]
      return {expected_groups[0]: torch.cat(stacked, dim=0)}
    tensor = _as_float_tensor(payload, device)
    return {expected_groups[0]: tensor}

  if isinstance(payload, np.ndarray):
    tensor = _as_float_tensor(payload, device)
    return {expected_groups[0]: tensor}

  if torch.is_tensor(payload):
    tensor = _as_float_tensor(payload, device)
    return {expected_groups[0]: tensor}

  raise TypeError(f"Unsupported observation payload type: {type(payload)!r}")


def _run_policy(policy, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
  with torch.inference_mode():
    actions = policy(inputs)
  if not torch.is_tensor(actions):
    raise TypeError(f"Policy returned unsupported output type: {type(actions)!r}")
  return actions.detach().cpu()


def _print_policy_overview(task_id: str, checkpoint_path: Path, policy) -> None:
  expected_groups = list(getattr(policy, "obs_groups", ["actor"]))
  obs_dim = getattr(policy, "obs_dim", None)
  action_dim = getattr(policy, "output_dim", None)
  print(f"[INFO] Task: {task_id}", file=sys.stderr)
  print(f"[INFO] Checkpoint: {checkpoint_path}", file=sys.stderr)
  print(f"[INFO] Expected observation groups: {expected_groups}", file=sys.stderr)
  if obs_dim is not None:
    print(f"[INFO] Expected flat obs dim: {obs_dim}", file=sys.stderr)
  if action_dim is not None:
    print(f"[INFO] Action dim: {action_dim}", file=sys.stderr)


def load_policy(task_id: str, cfg: InferConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  else:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    checkpoint_path = get_checkpoint_path(
      log_root_path,
      agent_cfg.load_run,
      agent_cfg.load_checkpoint,
    )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  env.close()

  _print_policy_overview(task_id, checkpoint_path, policy)
  return policy, checkpoint_path


def _infer_single(policy, payload: Any, device: str) -> torch.Tensor:
  inputs = _payload_to_inputs(payload, policy, device)
  expected_groups = list(getattr(policy, "obs_groups", ["actor"]))
  expected_dim = getattr(policy, "obs_dim", None)

  if len(expected_groups) == 1:
    group_name = expected_groups[0]
    flat_dim = inputs[group_name].shape[-1]
    if expected_dim is not None and flat_dim != expected_dim:
      raise ValueError(
        f"Observation dimension mismatch for '{group_name}': got {flat_dim}, expected {expected_dim}"
      )

  return _run_policy(policy, inputs)


def _print_actions(actions: torch.Tensor) -> None:
  if actions.ndim == 2 and actions.shape[0] == 1:
    payload: Any = actions[0].tolist()
  else:
    payload = actions.tolist()
  print(json.dumps(payload))


def run_infer(task_id: str, cfg: InferConfig) -> None:
  policy, _ = load_policy(task_id, cfg)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  if cfg.stream_stdin:
    for line in sys.stdin:
      line = line.strip()
      if not line:
        continue
      payload = json.loads(line)
      actions = _infer_single(policy, payload, device)
      _print_actions(actions)
    return

  if cfg.obs_file is None:
    raise ValueError("Provide --obs-file or enable --stream-stdin.")

  payload = _load_payload(Path(cfg.obs_file))
  actions = _infer_single(policy, payload, device)

  if cfg.output_file is not None:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".npy":
      np.save(output_path, actions.numpy())
    else:
      with output_path.open("w", encoding="utf-8") as handle:
        json.dump(actions.tolist(), handle)
  else:
    _print_actions(actions)


def main() -> None:
  maybe_print_top_level_help("infer")
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    InferConfig,
    args=remaining_args,
    default=InferConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_infer(chosen_task, args)


if __name__ == "__main__":
  main()