#!/usr/bin/env python3
"""Real-to-sim system identification of the SpiRob joint and tendon parameters.

The measured tendon forces are replayed into MuJoCo and the free model
parameters are fitted so the simulated joint angles follow the ArUco-tracked
real angles as closely as the model structure allows.

Two stages, in this order::

    # 1. seed the model from the free-vibration results, one forward sim,
    #    baseline cost -- always run this first, it tells you where you start
    uv run sysid/simulation_based/real2sim.py --mode validate

    # 2. optimize around those seeds
    uv run sysid/simulation_based/real2sim.py --mode finetune \
        --optimizer cma --maxiter 500 --workers 8

Cost function (see docs/sysid/simulation-based.md)::

    J = sqrt(sum_ij w_i e_pos^2) + lambda_vel * sqrt(sum_ij w_i e_vel^2)

with the angle error converted to degrees before squaring and
``lambda_vel = 0.05`` so the position trajectory dominates while damping stays
observable through the velocity term. An unstable simulation (NaN/Inf,
|qpos| > 1e3, MuJoCo's "huge value in QACC" warning) returns a finite penalty
of 1e6 instead of crashing, so the optimizer is steered away without NaN
corruption.

Parameter groups
----------------
``--optimize`` selects which groups are free; ``--represent`` selects how each
group is parameterized (``per_joint`` / ``shared`` / ``poly:N``); ``--band`` or
``--bounds`` sets the search range. See ``DEFAULT_PARAM_GROUPS`` below and
``GROUP_DOMAIN`` for the joint / tendon / opt / broadcast domains.

Joint order
-----------
Every per-joint array is in MuJoCo model-joint-index order: index 0 = base
joint (``j_12``), index njnt-1 = tip (``j_0``). The dataset's ``joint_1_deg``
is the base joint, so real ``joint_N`` maps to model index ``N-1``. Use
``--reverse-real-joints`` only for a dataset that numbers joints from the tip.

Ctrl+C during a finetune stops cleanly and still writes the best-so-far result.

Inputs : data/trajectories/*.parquet, models/spirob_13seg.xml,
         data/free_vibration/joint_*/sysid_settings.yaml (seeds)
Outputs: build/real2sim/*.{png,json,xml}
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np
import polars as pl
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import differential_evolution

from spirob.paths import (
    DEFAULT_MODEL,
    DEFAULT_TRAJECTORY,
    FREE_VIBRATION_DIR,
)
from spirob.paths import (
    build_dir as _build_dir,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ── Parameter identification / optimization configuration ───────────────────
# Per-joint seed values come from the free-vibration sys-id results stored in
# data/free_vibration/joint_*/sysid_settings.yaml (results.k_mean / results.d_mean).
IDENTIFIED_DIR = FREE_VIBRATION_DIR
FRICTIONLOSS_INIT = 0.15   # uniform seed for joint frictionloss (no measurement available)
VALUE_FLOOR = 1e-5         # lower clip for any per-element physical value (keeps the sim valid)
DEFAULT_SETTLING_STEPS = 1000  # quasi-static settling steps before each replay (see _SETTLING_STEPS)

# Broadcast solver knobs: a single scalar written across ALL joints/dofs of a
# model array column (they are set globally in the XML <default>). Each maps to
#   (model array attribute, column index or None for 1-D, default (lo, hi) bounds).
# solreflimit/solimplimit → joint-limit constraint (jnt_solref/jnt_solimp);
# solreffriction/solimpfriction → dry-friction constraint (dof_solref/dof_solimp);
# armature → added rotor inertia (dof_armature). solref = (timeconst, dampratio),
# solimp = (dmin, dmax, width, midpoint, power) — we expose the first three.
BROADCAST_KNOBS: dict[str, tuple[str, int | None, tuple[float, float]]] = {
    "armature":             ("dof_armature", None, (1e-4, 0.1)),
    "solreflimit_time":     ("jnt_solref", 0, (0.002, 0.2)),
    "solreflimit_damp":     ("jnt_solref", 1, (0.1, 5.0)),
    "solimplimit_dmin":     ("jnt_solimp", 0, (0.5, 0.99)),
    "solimplimit_dmax":     ("jnt_solimp", 1, (0.9, 0.9999)),
    "solimplimit_width":    ("jnt_solimp", 2, (1e-4, 0.02)),
    "solreffriction_time":  ("dof_solref", 0, (0.002, 0.2)),
    "solreffriction_damp":  ("dof_solref", 1, (0.1, 5.0)),
    "solimpfriction_dmin":  ("dof_solimp", 0, (0.5, 0.99)),
    "solimpfriction_dmax":  ("dof_solimp", 1, (0.9, 0.9999)),
    "solimpfriction_width": ("dof_solimp", 2, (1e-4, 0.02)),
}

# Optimizable parameter groups and their DOMAIN (number of values per group):
#   'joint'     → one value per joint  (njnt): jnt_stiffness / dof_damping / dof_frictionloss
#   'tendon'    → one value per tendon (ntendon): tendon_stiffness / tendon_damping / tendon_frictionloss
#   'opt'       → a single scalar written to model.opt.<name> (e.g. impratio;
#                 the group name MUST match the model.opt attribute)
#   'broadcast' → a single scalar written across all joints/dofs of a model
#                 array column (see BROADCAST_KNOBS: solref/solimp/armature)
# All seeds are read from the model/XML in build_seeds. The dict order is the
# order in which active groups are packed into the decision vector.
GROUP_DOMAIN = {
    "stiffness": "joint",
    "damping": "joint",
    "frictionloss": "joint",
    "tendon_stiffness": "tendon",
    "tendon_damping": "tendon",
    "tendon_frictionloss": "tendon",
    "impratio": "opt",
}
GROUP_DOMAIN.update({name: "broadcast" for name in BROADCAST_KNOBS})
GROUP_ORDER = tuple(GROUP_DOMAIN)


@dataclass
class ParamGroupConfig:
    """Configures one optimizable parameter group (joint- or tendon-domain).

    A group holds one value per element — per joint for joint-domain groups
    (njnt values), per tendon for tendon-domain groups (ntendon values), see
    GROUP_DOMAIN.

    representation:
      - 'per_joint': one free value per element (independent), searched in a
        relative ``band`` around the seed value (the name is kept for both domains).
      - 'shared':    a SINGLE free value applied to ALL elements of the group
        (e.g. both symmetric tendons get the same value). 1 parameter total.
      - 'poly':      value(element) = polyval(coeffs, x_norm), x_norm the
        normalized element index in [0, 1]. Only (poly_degree + 1) coefficients
        are optimized → fewer parameters, smooth distribution. (For 2-tendon
        groups a polynomial is degenerate; use 'shared' or 'per_joint'.)
    band: relative search half-width around the seed (finetune). 0.3 → ±30 %.
    bounds: optional absolute (min, max) search range for the value.
        When set it OVERRIDES ``band`` for this group — useful when the seed is
        tiny or zero (e.g. damping ~1e-3, tendon_damping=0) and a relative band
        would barely move it. Applies to both representations (poly output is
        clipped into the range).
    """
    name: str
    optimize: bool = True
    representation: str = "per_joint"   # 'per_joint' | 'poly'
    poly_degree: int = 3
    band: float = 0.3
    bounds: tuple[float, float] | None = None


# Edit this block to choose WHAT gets optimized and HOW. CLI flags
# (--optimize / --represent / --band / --bounds) can override it per run.
DEFAULT_PARAM_GROUPS: list[ParamGroupConfig] = [
    # ── joint groups (one value per joint) ──
    ParamGroupConfig("stiffness",    optimize=True, representation="poly", bounds=(0.1, 200.0)),  # representation="poly:<degree>", or representation="per_joint"
    ParamGroupConfig("damping",      optimize=True, representation="poly", bounds=(1e-4, 200.1)),  # or absolute: bounds=(1e-4, 0.1)
    # frictionloss is unmeasured (uniform seed); give it a wider band to explore.
    ParamGroupConfig("frictionloss", optimize=True, representation="poly", bounds=[0.0, 10.0]),  # or representation="poly" with band=0.3
    # ── tendon groups (seeds read from the XML) ──
    # 'shared' → both tendons get the SAME value (they are symmetric). Use
    # representation="per_joint" if you want the two tendons independent.
    ParamGroupConfig("tendon_stiffness",    optimize=True, representation="shared", bounds=(1.0, 1000.0)),
    ParamGroupConfig("tendon_damping",      optimize=True, representation="shared", bounds=(0.0, 200.0)),
    ParamGroupConfig("tendon_frictionloss", optimize=True, representation="shared", bounds=(0.0, 20.0)),
    # ── solver options (single scalar, written to model.opt.<name>) ──
    #ParamGroupConfig("impratio", optimize=True, representation="shared", bounds=(1.0, 100.0)),
]

# Broadcast solver knobs (off by default) — one global scalar each, seeded from
# the XML, bounds from BROADCAST_KNOBS. Enable per run with e.g.
#   --optimize solreflimit_time,solimplimit_dmin,armature
# or flip optimize=True here. Available names: see BROADCAST_KNOBS above
# (armature, solreflimit_time/damp, solimplimit_dmin/dmax/width,
#  solreffriction_time/damp, solimpfriction_dmin/dmax/width).
DEFAULT_PARAM_GROUPS += [
    ParamGroupConfig(name, optimize=True, representation="shared", bounds=bnd)
    for name, (_attr, _col, bnd) in BROADCAST_KNOBS.items()
]

# Globals filled during data prep
_GT_QPOS: np.ndarray | None = None
_GT_QVEL: np.ndarray | None = None
_SIM_TIMESTEPS: np.ndarray | None = None
_FORCE_INTERP_0 = None
_FORCE_INTERP_1 = None
_RECORD_DT: float = 0.0
# Actual simulation duration (may be shorter than requested sim_time if real data is shorter)
_ACTUAL_SIM_TIME: float = 0.0
# Real dataset joint numbering vs. MuJoCo model joint index.
# The tracked joint_1_deg is the BASE joint, which is model index 0 = j_12
# (also the base in the XML tree) — so NO column reversal is needed: real
# joint_N maps to model index N-1. Set True only if a dataset numbers joints
# from the tip instead. (Confirmed base=joint_1 with the user, 2026-07.)
_REVERSE_REAL_JOINT_ORDER: bool = False
_VIEWER_ENABLED: bool = False
_VIEWER_INTERVAL: int = 100
# Quasi-static settling steps run before every replay (biggest per-eval cost).
# 1000 @ dt=0.004 = 4 s of settling; lower it to speed up optimization.
_SETTLING_STEPS: int = DEFAULT_SETTLING_STEPS
# first-call print flags
_FIRST_SIM_PRINTED = False
_FIRST_COST_PRINTED = False
_FIRST_COST_MATCH_PRINTED = False
_EVAL_COUNTER = 0


def _maybe_reverse_joint_order(arr_2d: np.ndarray, reverse_order: bool) -> np.ndarray:
    """Reverse joint axis (columns) if requested.

    Centralized place for real<->simulation joint index mapping.
    """
    if not reverse_order:
        return arr_2d
    return arr_2d[:, ::-1]


def _save_signal_comparison_plot(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    y_filtered: np.ndarray,
    title: str,
    y_label: str,
    out_path: Path,
    legend_label: str,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_raw, y_raw, color="lightgray", linewidth=1.0, alpha=0.8, label="raw")
    ax.plot(x_raw[: len(y_filtered)], y_filtered, color="steelblue", linewidth=1.6, label="filtered")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def detect_and_fill_outliers(data_1d: np.ndarray, thresh: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using MAD-based z-score and fill them by interpolation.

    Returns (filled_array, mask_outliers) where mask_outliers is boolean array True for outliers.
    """
    arr = data_1d.copy()
    if arr.size == 0:
        return arr, np.zeros_like(arr, dtype=bool)

    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad < 1e-12:
        # If MAD is zero (flat signal), treat no outliers
        return arr, np.zeros_like(arr, dtype=bool)

    z = 0.6745 * np.abs(arr - median) / mad
    mask = z > thresh

    if not np.any(mask):
        return arr, mask

    # Indices of valid (non-outlier) points
    idx = np.arange(arr.size)
    valid_idx = idx[~mask]
    valid_vals = arr[~mask]

    if valid_idx.size == 0:
        # all outliers: fallback to median
        filled = np.full_like(arr, median)
        return filled, mask

    if valid_idx.size == 1:
        # single valid point: fill with that value
        filled = np.full_like(arr, valid_vals[0])
        return filled, mask

    # interpolate across valid points; np.interp handles multiple consecutive outliers
    filled = arr.copy()
    filled[mask] = np.interp(idx[mask], valid_idx, valid_vals)
    return filled, mask


def detect_and_fill_outliers_simple(data_1d: np.ndarray, abs_thresh: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Simple outlier detection: mark a sample as outlier if it deviates from
    the average of its immediate neighbors by more than `abs_thresh`.

    This is intended for isolated spikes; multiple consecutive outliers are
    filled by interpolation across valid points (same strategy as MAD-based).
    Returns (filled_array, mask_outliers).
    """
    arr = data_1d.copy()
    n = arr.size
    if n == 0:
        return arr, np.zeros_like(arr, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    if n == 1:
        return arr, mask

    # interior points: compare to mean of neighbors
    for i in range(1, n - 1):
        neighbor_mean = 0.5 * (arr[i - 1] + arr[i + 1])
        if abs(arr[i] - neighbor_mean) > abs_thresh:
            mask[i] = True

    # endpoints: compare to the single neighbor
    if abs(arr[0] - arr[1]) > abs_thresh:
        mask[0] = True
    if abs(arr[-1] - arr[-2]) > abs_thresh:
        mask[-1] = True

    if not np.any(mask):
        return arr, mask

    idx = np.arange(n)
    valid_idx = idx[~mask]
    valid_vals = arr[~mask]

    if valid_idx.size == 0:
        filled = np.full_like(arr, np.median(arr))
        return filled, mask
    if valid_idx.size == 1:
        filled = np.full_like(arr, valid_vals[0])
        return filled, mask

    filled = arr.copy()
    filled[mask] = np.interp(idx[mask], valid_idx, valid_vals)
    return filled, mask


def downsample_uniform(
    data_array: np.ndarray,
    time_array: np.ndarray,
    record_dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample data uniformly by recording every N-th sample based on record_dt.

    This ensures data points are removed evenly across the entire dataset
    (not just truncated at the end). Uses the actual sampling rate of
    the data to compute record_every.

    Args:
        data_array: shape (N, ...) — array to downsample.
        time_array: shape (N,) — timestamps corresponding to data.
        record_dt: target recording interval (e.g. 0.1 s).

    Returns:
        (downsampled_data, downsampled_time) with shape (M, ...) where M << N.
    """
    if len(time_array) < 2 or record_dt <= 0:
        return data_array, time_array

    # Compute mean sampling period from actual time differences
    dt_actual = np.mean(np.diff(time_array))
    if dt_actual <= 0:
        return data_array, time_array

    # How many samples to skip between recordings
    record_every = max(1, int(round(record_dt / dt_actual)))

    # Uniformly select indices: 0, record_every, 2*record_every, ...
    indices = np.arange(0, len(time_array), record_every)

    downsampled_data = data_array[indices]
    downsampled_time = time_array[indices]

    return downsampled_data, downsampled_time


def save_preprocessing_plots(
    raw_time: np.ndarray,
    raw_force_0: np.ndarray,
    raw_force_1: np.ndarray,
    filtered_force_0: np.ndarray,
    filtered_force_1: np.ndarray,
    raw_qpos: np.ndarray,
    filtered_qpos: np.ndarray,
    raw_qvel: np.ndarray,
    filtered_qvel: np.ndarray,
    out_dir: Path,
) -> None:
    if plt is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    axs[0].plot(raw_time, raw_force_0, color="lightgray", linewidth=1.0, alpha=0.8, label="force_0 raw")
    axs[0].plot(raw_time[: len(filtered_force_0)], filtered_force_0, color="steelblue", linewidth=1.6, label="force_0 filtered")
    axs[0].plot(raw_time, raw_force_1, color="silver", linewidth=1.0, alpha=0.7, label="force_1 raw")
    axs[0].plot(raw_time[: len(filtered_force_1)], filtered_force_1, color="darkorange", linewidth=1.6, label="force_1 filtered")
    axs[0].set_title("Forces before/after filtering")
    axs[0].set_ylabel("force (N)")
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].legend(loc="best")

    for i in range(raw_qpos.shape[1]):
        axs[1].plot(raw_time, raw_qpos[:, i], color="lightgray", linewidth=0.9, alpha=0.5)
        axs[1].plot(raw_time[: len(filtered_qpos)], filtered_qpos[:, i], linewidth=1.3)
    axs[1].set_title("Joint positions before/after filtering")
    axs[1].set_ylabel("qpos (rad)")
    axs[1].grid(True, linestyle="--", alpha=0.4)

    for i in range(raw_qvel.shape[1]):
        axs[2].plot(raw_time, raw_qvel[:, i], color="lightgray", linewidth=0.9, alpha=0.5)
        axs[2].plot(raw_time[: len(filtered_qvel)], filtered_qvel[:, i], linewidth=1.3)
    axs[2].set_title("Joint velocities before/after filtering")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("qvel (rad/s)")
    axs[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "preprocessing_overview.png", dpi=150)
    plt.close(fig)

    _save_signal_comparison_plot(
        raw_time,
        raw_force_0,
        filtered_force_0,
        "Force 0 before/after filtering",
        "force (N)",
        out_dir / "force_0_comparison.png",
        "force 0",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_force_1,
        filtered_force_1,
        "Force 1 before/after filtering",
        "force (N)",
        out_dir / "force_1_comparison.png",
        "force 1",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_qpos[:, 0],
        filtered_qpos[:, 0],
        "qpos[0] before/after filtering",
        "qpos (rad)",
        out_dir / "qpos_0_comparison.png",
        "qpos 0",
    )
    _save_signal_comparison_plot(
        raw_time,
        raw_qvel[:, 0],
        filtered_qvel[:, 0],
        "qvel[0] before/after filtering",
        "qvel (rad/s)",
        out_dir / "qvel_0_comparison.png",
        "qvel 0",
    )
    # Also save per-joint comparison plots so each joint can be inspected individually
    nj = raw_qpos.shape[1]
    for j in range(nj):
        _save_signal_comparison_plot(
            raw_time,
            raw_qpos[:, j],
            filtered_qpos[:, j],
            f"qpos[{j}] before/after filtering",
            "qpos (rad)",
            out_dir / f"qpos_{j}_comparison.png",
            f"qpos {j}",
        )
        _save_signal_comparison_plot(
            raw_time,
            raw_qvel[:, j],
            filtered_qvel[:, j],
            f"qvel[{j}] before/after filtering",
            "qvel (rad/s)",
            out_dir / f"qvel_{j}_comparison.png",
            f"qvel {j}",
        )


def make_model(xml_file: Path | None = None) -> mj.MjModel:
    if xml_file is None:
        xml_file = DEFAULT_MODEL
    return mj.MjModel.from_xml_path(str(xml_file))


def set_params(model: mj.MjModel, params: dict[str, np.ndarray]) -> None:
    """Write all parameter groups from ``params`` into the model.

    ``params`` is keyed by group name (see GROUP_ORDER). Joint groups have njnt
    values, tendon groups have ntendon values, opt groups a single scalar. A
    missing group is left at the model's current value.
      stiffness    → jnt_stiffness[i]
      damping      → dof_damping[dofadr[i]]
      frictionloss → dof_frictionloss[dofadr[i]]   (Coulomb-like joint friction)
      tendon_stiffness / tendon_damping / tendon_frictionloss → tendon_*[t]
      impratio (and any 'opt' group) → model.opt.<name>
    """
    st = params.get("stiffness")
    dp = params.get("damping")
    fl = params.get("frictionloss")
    for i in range(model.njnt):
        dof = int(model.jnt_dofadr[i])
        if st is not None:
            model.jnt_stiffness[i] = float(st[i])
        if dp is not None:
            model.dof_damping[dof] = float(dp[i])
        if fl is not None:
            model.dof_frictionloss[dof] = float(fl[i])

    tst = params.get("tendon_stiffness")
    tdp = params.get("tendon_damping")
    tfl = params.get("tendon_frictionloss")
    for t in range(model.ntendon):
        if tst is not None:
            model.tendon_stiffness[t] = float(tst[t])
        if tdp is not None:
            model.tendon_damping[t] = float(tdp[t])
        if tfl is not None:
            model.tendon_frictionloss[t] = float(tfl[t])

    # solver-option ('opt' domain) params: scalar written to model.opt.<name>.
    # NOTE: this runs after the loops above and simulate_and_sample only sets
    # timestep/iterations, so the value we set here is what the solver uses.
    for name, domain in GROUP_DOMAIN.items():
        if domain == "opt" and name in params:
            setattr(model.opt, name, float(np.asarray(params[name]).ravel()[0]))

    # broadcast knobs: one scalar written across all joints/dofs of an array column
    for name, (attr, col, _bnd) in BROADCAST_KNOBS.items():
        if name in params:
            v = float(np.asarray(params[name]).ravel()[0])
            arr = getattr(model, attr)
            if col is None:
                arr[:] = v
            else:
                arr[:, col] = v


def get_local_model_and_data(xml_file: Path | None = None):
    # Create model+data on demand; safe for multiprocessing workers
    model = make_model(xml_file)
    data = mj.MjData(model)
    return model, data


# ============================================================================
# IDENTIFIED PARAMETER SEEDS  (free-vibration sys-id → per-joint k / d)
# ============================================================================

def _maybe_reverse_1d(arr: np.ndarray, reverse_order: bool) -> np.ndarray:
    """Reverse a 1-D per-joint array to match the same real<->sim joint mapping
    used for qpos columns (see ``_maybe_reverse_joint_order``)."""
    return arr[::-1] if reverse_order else arr


def load_identified_params(builds_dir: Path = IDENTIFIED_DIR) -> dict[int, dict]:
    """Read identified stiffness/damping from every ``sysid_settings.yaml``.

    Each ``data/free_vibration/joint_NN/sysid_settings.yaml`` (written by the
    free-vibration GUI) carries ``results.k_mean`` / ``results.d_mean`` for real
    joint ``NN``.

    Returns {real_joint_number: {'k': float, 'd': float}}.
    """
    anchors: dict[int, dict] = {}
    for yaml_path in sorted(builds_dir.glob("joint_*/sysid_settings.yaml")):
        folder = yaml_path.parent.name  # 'joint_NN'
        try:
            n = int(folder.rsplit("_", 1)[-1])
        except ValueError:
            continue
        try:
            with open(yaml_path) as f:
                doc = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"[identified] could not read {yaml_path}: {exc}")
            continue
        res = doc.get("results") or {}
        k = res.get("k_mean")
        d = res.get("d_mean")
        if k is None or d is None:
            continue
        anchors[n] = {"k": float(k), "d": float(d)}
    return anchors


def interpolate_joint_params(
    anchors: dict[int, dict], njnt: int, reverse_order: bool
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Interpolate anchor k/d (keyed by real joint number) across all joints,
    then map to simulation joint order.

    Anchor joint numbers are CLAMPED into [1, njnt] rather than dropped, so a
    dataset that numbers one joint beyond the model still lands on the tip.
    Linear interpolation between anchors, constant extrapolation at the ends.
    Returns (k_sim, d_sim, used_real_joints).
    """
    # Clamp each anchor into range; on collision the higher original number wins.
    by_pos: dict[int, tuple[int, dict]] = {}
    for n, v in sorted(anchors.items()):
        pos = min(max(int(n), 1), njnt)
        by_pos[pos] = (n, v)
    if not by_pos:
        raise RuntimeError(f"No identified anchors found (anchors={sorted(anchors)})")

    positions = sorted(by_pos)
    xp = np.array([p - 1 for p in positions], dtype=float)  # 0-based real index
    k_fp = np.array([by_pos[p][1]["k"] for p in positions], dtype=float)
    d_fp = np.array([by_pos[p][1]["d"] for p in positions], dtype=float)

    idx = np.arange(njnt, dtype=float)
    k_real = np.interp(idx, xp, k_fp)   # np.interp clamps (constant) outside xp
    d_real = np.interp(idx, xp, d_fp)

    k_sim = _maybe_reverse_1d(k_real, reverse_order)
    d_sim = _maybe_reverse_1d(d_real, reverse_order)
    used = [by_pos[p][0] for p in positions]  # original anchor numbers actually used
    return k_sim, d_sim, used


def build_seeds(
    model: mj.MjModel, reverse_order: bool, frictionloss_init: float = FRICTIONLOSS_INIT
) -> tuple[dict[str, np.ndarray], dict[int, dict], list[int]]:
    """Build seed arrays for all groups.

    Joint groups: stiffness/damping come from the identified anchors, frictionloss
    is seeded uniformly. Tendon groups and opt groups (e.g. impratio) are seeded
    from the model's current values (i.e. the XML). Returns (seeds, anchors,
    used_joints).
    """
    njnt = model.njnt
    anchors = load_identified_params()
    k_sim, d_sim, used = interpolate_joint_params(anchors, njnt, reverse_order)
    seeds = {
        "stiffness": np.asarray(k_sim, dtype=float),
        "damping": np.asarray(d_sim, dtype=float),
        "frictionloss": np.full(njnt, float(frictionloss_init), dtype=float),
        "tendon_stiffness": np.array(model.tendon_stiffness, dtype=float).copy(),
        "tendon_damping": np.array(model.tendon_damping, dtype=float).copy(),
        "tendon_frictionloss": np.array(model.tendon_frictionloss, dtype=float).copy(),
    }
    # opt-domain groups: single scalar read from model.opt.<name>
    for name, domain in GROUP_DOMAIN.items():
        if domain == "opt":
            seeds[name] = np.array([float(getattr(model.opt, name))], dtype=float)
    # broadcast knobs: single scalar read from the model array column (mean)
    for name, (attr, col, _bnd) in BROADCAST_KNOBS.items():
        arr = np.asarray(getattr(model, attr), dtype=float)
        cur = arr if col is None else arr[:, col]
        seeds[name] = np.array([float(np.mean(cur))], dtype=float)
    return seeds, anchors, used


# ── CANONICAL JOINT ORDER (single source of truth) ──────────────────────────
# EVERY per-joint array in this module — seeds, qpos/qvel, the stiffness/
# damping/frictionloss vectors, and the optimizer output — is ordered by the
# MuJoCo *model joint index* i (0 … njnt-1). set_params() writes value[i] into
# model joint i, and _GT_QPOS[:, i] is that joint's measured angle. The real
# dataset column joint_N_deg maps to model index i via _maybe_reverse_joint_order
# (reverse=True → model index i ↔ real joint (njnt - i); reverse=False ↔ i+1).
# joint_index_labels() is the ONLY place that spells this mapping out; use it
# anywhere a human/file needs to know which value belongs to which joint.

def joint_index_labels(model: mj.MjModel, reverse_order: bool) -> list[dict]:
    """Return, per model joint index i, its model joint name and the real
    dataset joint number (joint_N_deg) that maps to it.

    This is the authoritative index→joint mapping for all per-joint arrays.
    """
    njnt = model.njnt
    labels = []
    for i in range(njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) or f"model_joint_{i}"
        real = (njnt - i) if reverse_order else (i + 1)
        labels.append({"model_index": i, "model_joint": name, "real_joint": int(real)})
    return labels


# ============================================================================
# MODULAR PARAMETER SPACE  (per-joint values  <->  optimizer vector)
# ============================================================================

@dataclass
class GroupSpec:
    """Resolved layout of one active group inside the optimizer vector."""
    name: str
    representation: str
    slice: tuple[int, int]     # [start, end) indices into x
    xn: np.ndarray             # normalized joint coordinates in [0, 1]
    seed: np.ndarray           # per-joint seed values (sim order)
    clip_lo: float = VALUE_FLOOR   # decoded per-joint values are clipped to
    clip_hi: float = np.inf        # [clip_lo, clip_hi] (enforces absolute bounds)


def _poly_seed_coeffs(seed: np.ndarray, xn: np.ndarray, degree: int) -> np.ndarray:
    """Highest-degree-first coefficients fitting ``seed`` over ``xn``."""
    degree = int(min(degree, len(seed) - 1))
    return np.polyfit(xn, seed, degree)


def build_param_space(
    groups: list[ParamGroupConfig], seeds: dict[str, np.ndarray]
) -> tuple[list[GroupSpec], np.ndarray, list[tuple[float, float]]]:
    """Pack the active groups into (specs, x0, bounds).

    Each group's element count comes from its seed length (njnt for joint groups,
    ntendon for tendon groups). Each group uses either a relative ``band`` around
    the seed or an absolute ``bounds=(min, max)`` range (bounds wins when set):

    per_joint, band:    value bounds = [(1-band)*seed, (1+band)*seed].
    per_joint, bounds:  value bounds = [min, max]; seed clipped into range as x0.
    poly, band:         (deg+1) coeffs seeded via polyfit; each bounded by
                        ±band*(|c| + cref) so near-zero coefficients still move.
    poly, bounds:       coeffs fit the clipped seed, each bounded by ±(max-min);
                        the polynomial output is clipped to [min, max] on decode.
    """
    specs: list[GroupSpec] = []
    x0_parts: list[np.ndarray] = []
    bounds: list[tuple[float, float]] = []
    cursor = 0

    for g in groups:
        if not g.optimize:
            continue
        seed = np.asarray(seeds[g.name], dtype=float)
        nel = len(seed)                          # njnt or ntendon
        xn = np.linspace(0.0, 1.0, nel)

        abs_bounds = g.bounds is not None
        if abs_bounds:
            b_lo, b_hi = float(g.bounds[0]), float(g.bounds[1])
            if not b_hi > b_lo:
                raise ValueError(f"{g.name}: bounds min must be < max, got {g.bounds}")
            clip_lo = max(VALUE_FLOOR, b_lo)
            clip_hi = b_hi
        else:
            clip_lo, clip_hi = VALUE_FLOOR, np.inf

        if g.representation == "poly":
            if abs_bounds:
                seed_fit = np.clip(seed, b_lo, b_hi)
                coeffs = _poly_seed_coeffs(seed_fit, xn, g.poly_degree)
                span = b_hi - b_lo
                lo = coeffs - span
                hi = coeffs + span
            else:
                coeffs = _poly_seed_coeffs(seed, xn, g.poly_degree)
                cref = 0.1 * float(np.mean(np.abs(seed))) + 1e-9
                lo = coeffs - g.band * (np.abs(coeffs) + cref)
                hi = coeffs + g.band * (np.abs(coeffs) + cref)
            n = len(coeffs)
            x0_parts.append(coeffs)
        elif g.representation == "per_joint":
            n = nel
            if abs_bounds:
                lo = np.full(nel, b_lo)
                hi = np.full(nel, b_hi)
                x0_parts.append(np.clip(seed, b_lo, b_hi))
            else:
                lo = np.maximum((1.0 - g.band) * seed, VALUE_FLOOR)
                hi = np.maximum((1.0 + g.band) * seed, lo + VALUE_FLOOR)
                x0_parts.append(seed.copy())
        elif g.representation == "shared":
            # One free value applied to ALL elements (e.g. both tendons equal).
            n = 1
            s = float(np.mean(seed))
            if abs_bounds:
                lo = np.array([b_lo])
                hi = np.array([b_hi])
                x0_parts.append(np.array([float(np.clip(s, b_lo, b_hi))]))
            else:
                lo_s = max((1.0 - g.band) * s, VALUE_FLOOR)
                hi_s = max((1.0 + g.band) * s, lo_s + VALUE_FLOOR)
                lo = np.array([lo_s])
                hi = np.array([hi_s])
                x0_parts.append(np.array([s]))
        else:
            raise ValueError(
                f"Unknown representation '{g.representation}' for {g.name} "
                f"(use per_joint | shared | poly)"
            )

        specs.append(GroupSpec(g.name, g.representation, (cursor, cursor + n), xn, seed,
                               clip_lo=clip_lo, clip_hi=clip_hi))
        for a, b in zip(np.atleast_1d(lo), np.atleast_1d(hi)):
            bounds.append((float(a), float(b)))
        cursor += n

    x0 = np.concatenate(x0_parts) if x0_parts else np.array([], dtype=float)
    return specs, x0, bounds


def decode_params(
    x: np.ndarray, specs: list[GroupSpec], seeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Reconstruct physical arrays for ALL groups (joint- and tendon-domain).

    Active groups are decoded from ``x`` (their representation); inactive groups
    fall back to their seed. Values are clipped to each group's [clip_lo, clip_hi]
    (a floor for band mode, the absolute range for bounds mode).
    """
    out = {name: np.asarray(seeds[name], dtype=float).copy() for name in GROUP_ORDER}
    for spec in specs:
        a, b = spec.slice
        part = x[a:b]
        if spec.representation == "poly":
            vals = np.polyval(part, spec.xn)
        elif spec.representation == "shared":
            vals = np.full(len(spec.seed), float(part[0]))  # broadcast the single value
        else:
            vals = part
        out[spec.name] = np.clip(np.asarray(vals, dtype=float), spec.clip_lo, spec.clip_hi)
    return out


def should_preview_simulation(eval_index: int, preview_enabled: bool, preview_interval: int) -> bool:
    """Return True when the current evaluation should be shown in the passive viewer.

    Previewing is intentionally opt-in. The first evaluation is always shown,
    then every `preview_interval` evaluations.
    """
    if not preview_enabled:
        return False
    if eval_index <= 1:
        return True
    if preview_interval <= 0:
        return False
    return eval_index % preview_interval == 0


def preview_simulation_passive(
    model: mj.MjModel,
    qpos_init: np.ndarray,
    sim_time: float,
    record_dt: float = 0.0,
    title: str = "Simulation preview",
) -> None:
    """Show one replay pass in a passive MuJoCo viewer in real time.

    This is separate from the optimizer logic on purpose: when the viewer is
    disabled, this function is never called and does nothing to the run time.
    """
    if not _VIEWER_ENABLED:
        return
    if mj_viewer is None:
        return
    if model.nu < 2:
        return

    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:] = qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    dt = model.opt.timestep
    effective_sim_time = _ACTUAL_SIM_TIME if _ACTUAL_SIM_TIME > 0 else sim_time
    total_steps = max(1, int(effective_sim_time / dt))
    print(f"[viewer] {title}: real-time replay for {effective_sim_time:.3f}s at dt={dt:.4f}s")

    with mj_viewer.launch_passive(model, data) as viewer:
        step_index = 0
        while viewer.is_running() and step_index <= total_steps:
            step_start = time.time()
            apply_measured_forces(data, data.time)
            mj.mj_step(model, data)

            viewer.sync()
            dt_left = dt - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

            step_index += 1


def apply_measured_forces(data: mj.MjData, t: float) -> tuple[float, float]:
    """Write the measured force profile into `data.ctrl` at simulation time `t`.

    `t` is MuJoCo's current simulation time. We use it only as the lookup key
    for the interpolated force signals; the controller itself is just replaying
    measured inputs.
    """
    if _FORCE_INTERP_0 is None or _FORCE_INTERP_1 is None:
        return 0.0, 0.0

    force_0 = float(_FORCE_INTERP_0(t))
    force_1 = float(_FORCE_INTERP_1(t))
    data.ctrl[0] = -force_0
    data.ctrl[1] = -force_1
    return force_0, force_1


def simulate_and_sample(model: mj.MjModel, data: mj.MjData, sim_time: float, qpos_init: np.ndarray, record_dt: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mj.mj_resetData(model, data)
    model.opt.timestep = 0.004
    model.opt.iterations = 20

    if model.nu < 2:
        raise RuntimeError(f"Expected at least 2 actuators for force replay, got model.nu={model.nu}")

    data.qpos[:] = qpos_init
    data.qvel[:] = np.zeros(model.nv)
    mj.mj_forward(model, data)

    # Settling phase: let the system calm down under the applied forces.
    # Apply the same interpolated force profile during settling so the system
    # reaches a force-induced equilibrium before we reset time and replay.
    settling_steps = _SETTLING_STEPS
    for _ in range(settling_steps):
        apply_measured_forces(data, data.time)
        mj.mj_step(model, data)
        data.qvel[:] = 0.0

    # After settling, restart trajectory time so the main simulation replays the
    # measured force profile from t=0 against a settled initial state.
    data.time = 0.0
    dt = model.opt.timestep
    # Use actual_sim_time if available (from preprocessing); fallback to sim_time
    effective_sim_time = _ACTUAL_SIM_TIME if _ACTUAL_SIM_TIME > 0 else sim_time
    total_steps = max(1, int(effective_sim_time / dt))

    # Determine recording interval: if record_dt > 0, record every N steps; else record all
    record_every = max(1, int(round(record_dt / dt))) if record_dt > 0 else 1

    qpos_hist = []
    qvel_hist = []
    applied_f0_hist = []
    applied_f1_hist = []
    for step in range(total_steps + 1):
        if step % record_every == 0:
            qpos_hist.append(data.qpos.copy())
            qvel_hist.append(data.qvel.copy())
            # record the force values applied at this simulation time (positive N)
            force_0, force_1 = apply_measured_forces(data, data.time)
            applied_f0_hist.append(force_0)
            applied_f1_hist.append(force_1)
        else:
            apply_measured_forces(data, data.time)
        mj.mj_step(model, data)

    qpos_arr = np.array(qpos_hist)
    qvel_arr = np.array(qvel_hist)
    applied_f0_arr = np.array(applied_f0_hist)
    applied_f1_arr = np.array(applied_f1_hist)

    # construct recorded time array for returned samples
    sim_steps = np.arange(0, total_steps + 1)
    record_indices = sim_steps[::record_every]
    sim_times = record_indices * dt

    # Print shapes once to help debugging and verification
    global _FIRST_SIM_PRINTED
    if not _FIRST_SIM_PRINTED:
        try:
            print(f"[simulate] produced est_qpos shape={qpos_arr.shape}, est_qvel shape={qvel_arr.shape}, record_dt={record_dt}, total_steps={total_steps}, record_every={record_every}, effective_sim_time={effective_sim_time:.4f}s")
        except Exception:
            pass
        _FIRST_SIM_PRINTED = True

    return qpos_arr, qvel_arr, applied_f0_arr, applied_f1_arr, sim_times


def load_and_preprocess(path: Path, sim_time: float, force_window: int = 60, data_smooth_window: int = 60, outlier_thresh: float = 0.0005, simple_outlier: bool = False, outlier_abs_thresh: float = 0.1, record_dt: float = 0.0):
    global _GT_QPOS, _GT_QVEL, _SIM_TIMESTEPS, _FORCE_INTERP_0, _FORCE_INTERP_1
    if not path.exists():
        raise FileNotFoundError(path)

    df = pl.read_parquet(path)
    joint_cols = [f"joint_{i}_deg" for i in range(1, 14)]
    df = df.drop_nulls(subset=["global_timestamp_s", "meas_force_0_N", "meas_force_1_N"] + joint_cols)
    if len(df) == 0:
        raise RuntimeError("No valid rows after dropping nulls")

    raw_time = df["global_timestamp_s"].to_numpy()
    raw_time = raw_time - raw_time[0]

    f0_raw = df["meas_force_0_N"].to_numpy()
    f1_raw = df["meas_force_1_N"].to_numpy()
    # Outlier detection: choose simple neighbor-based or MAD-based
    if simple_outlier:
        f0_filled, f0_mask = detect_and_fill_outliers_simple(f0_raw, abs_thresh=outlier_abs_thresh)
        f1_filled, f1_mask = detect_and_fill_outliers_simple(f1_raw, abs_thresh=outlier_abs_thresh)
    else:
        f0_filled, f0_mask = detect_and_fill_outliers(f0_raw, thresh=outlier_thresh)
        f1_filled, f1_mask = detect_and_fill_outliers(f1_raw, thresh=outlier_thresh)
    f0 = uniform_filter1d(f0_filled, size=force_window)
    f1 = uniform_filter1d(f1_filled, size=force_window)

    qpos_raw = np.zeros((len(df), 13))
    for i, col in enumerate(joint_cols):
        rads = np.deg2rad(df[col].to_numpy())
        qpos_raw[:, i] = rads

    # Apply real->simulation joint index mapping exactly once in preprocessing.
    qpos_raw_before_map = qpos_raw.copy()
    qpos_raw = _maybe_reverse_joint_order(qpos_raw, _REVERSE_REAL_JOINT_ORDER)
    print(f"Joint order mapping: reverse_real_joint_order={_REVERSE_REAL_JOINT_ORDER}")
    if len(qpos_raw) > 0:
        print(
            "[joint_map] sample t0 first3 before->after: "
            f"{np.array2string(qpos_raw_before_map[0, :3], precision=4)} -> "
            f"{np.array2string(qpos_raw[0, :3], precision=4)}"
        )

    # Per-joint outlier detection/filling (simple or MAD)
    qpos_filled = np.zeros_like(qpos_raw)
    qpos_masks = np.zeros_like(qpos_raw, dtype=bool)
    for i in range(qpos_raw.shape[1]):
        if simple_outlier:
            filled, mask = detect_and_fill_outliers_simple(qpos_raw[:, i], abs_thresh=outlier_abs_thresh)
        else:
            filled, mask = detect_and_fill_outliers(qpos_raw[:, i], thresh=outlier_thresh)
        qpos_filled[:, i] = filled
        qpos_masks[:, i] = mask

    # then smooth the filled signals
    qpos = np.zeros_like(qpos_raw)
    for i in range(qpos_raw.shape[1]):
        qpos[:, i] = uniform_filter1d(qpos_filled[:, i], size=data_smooth_window)

    qvel_raw = np.zeros_like(qpos_raw)
    if len(raw_time) > 1:
        for i in range(qpos_raw.shape[1]):
            # compute from raw/fill-equals-raw positions for inspection
            qvel_raw[:, i] = np.gradient(qpos_filled[:, i], raw_time)

    max_t = raw_time[-1] if sim_time <= 0 else min(sim_time, raw_time[-1])
    mask = raw_time <= max_t
    t_target = raw_time[mask]
    qpos_target = qpos[mask]
    f0_target = f0[mask]
    f1_target = f1[mask]

    # Store the actual simulation duration for use in cost function and validation
    global _ACTUAL_SIM_TIME
    _ACTUAL_SIM_TIME = max_t

    qvel_target = np.zeros_like(qpos_target)
    if len(t_target) > 1:
        for i in range(qpos_target.shape[1]):
            raw_vel = np.gradient(qpos[mask, i], t_target)
            # also smooth velocities
            qvel_target[:, i] = uniform_filter1d(raw_vel, size=data_smooth_window)

    # ── Uniform downsampling (if record_dt > 0) ────────────────────────────────────────
    # This removes data points evenly across the entire dataset, not just truncating at end.
    # Critical: ensures real-data samples align with simulation samples (both use same record_dt).
    if record_dt > 0:
        # Use one shared index selection for all signals to preserve strict alignment.
        if len(t_target) > 1:
            dt_actual = np.mean(np.diff(t_target))
            if dt_actual > 0:
                record_every = max(1, int(round(record_dt / dt_actual)))
                sel_idx = np.arange(0, len(t_target), record_every)
            else:
                sel_idx = np.arange(len(t_target))
        else:
            sel_idx = np.arange(len(t_target))

        t_target = t_target[sel_idx]
        qpos_target = qpos_target[sel_idx]
        qvel_target = qvel_target[sel_idx]
        f0_target = f0_target[sel_idx]
        f1_target = f1_target[sel_idx]

        # Ensure final lengths match what the simulation will record.
        # Compute expected simulated sample count from sim_time and record_dt
        if sim_time > 0:
            expected_sim_samples = int(np.floor(sim_time / record_dt)) + 1
        else:
            expected_sim_samples = len(t_target)

        if expected_sim_samples != len(t_target):
            final_n = min(expected_sim_samples, len(t_target))
            sel_idx = np.round(np.linspace(0, len(t_target) - 1, final_n)).astype(int)
            t_target = t_target[sel_idx]
            qpos_target = qpos_target[sel_idx]
            qvel_target = qvel_target[sel_idx]
            f0_target = f0_target[sel_idx]
            f1_target = f1_target[sel_idx]

    save_preprocessing_plots(
        raw_time,
        f0_raw,
        f1_raw,
        f0,
        f1,
        qpos_raw,
        qpos,
        qvel_raw,
        qvel_target,
        _build_dir("preprocessing_plots"),
    )

    _GT_QPOS = qpos_target
    _GT_QVEL = qvel_target
    _SIM_TIMESTEPS = t_target
    _FORCE_INTERP_0 = interp1d(t_target, f0_target, kind='linear', fill_value=(f0_target[0], f0_target[-1]), bounds_error=False)
    _FORCE_INTERP_1 = interp1d(t_target, f1_target, kind='linear', fill_value=(f1_target[0], f1_target[-1]), bounds_error=False)

    return {
        "t": t_target,
        "qpos": qpos_target,
        "qvel": qvel_target,
        "f0": f0_target,
        "f1": f1_target,
    }


def resample_match(gt: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly resample the longer of gt/est onto the shorter one's length."""
    lg, le = len(gt), len(est)
    if lg == le:
        return gt, est
    if lg > le:
        idx = np.round(np.linspace(0, lg - 1, le)).astype(int)
        return gt[idx], est
    idx = np.round(np.linspace(0, le - 1, lg)).astype(int)
    return gt, est[idx]


def compute_cost(
    gt_qpos: np.ndarray, gt_qvel: np.ndarray, est_qpos: np.ndarray, est_qvel: np.ndarray
) -> float:
    """Time-weighted RMSE of position (deg) + 0.5·velocity (deg/s).

    Length mismatches between real and simulated samples are reconciled by
    uniform resampling. Returns a large penalty on any shape mismatch.
    """
    if len(gt_qpos) == 0 or len(est_qpos) == 0:
        return 1e6
    gt_p, est_p = resample_match(gt_qpos, est_qpos)
    gt_v, est_v = resample_match(gt_qvel, est_qvel)
    if gt_p.shape != est_p.shape or gt_v.shape != est_v.shape:
        return 1e6

    # Instability guard: an unstable sim yields NaN/Inf or blown-up values (the
    # "Nan, Inf or huge value in QACC … simulation is unstable" warning). Return
    # a large finite penalty so DE/CMA steer away from that region (never NaN,
    # which would corrupt the optimizer).
    if not (np.all(np.isfinite(est_p)) and np.all(np.isfinite(est_v))):
        return 1e6
    if np.max(np.abs(est_p)) > 1e3:  # joint angles never legitimately get this large
        return 1e6

    err_pos_deg = np.rad2deg(gt_p - est_p)
    err_vel_deg = np.rad2deg(gt_v - est_v)
    n = len(err_pos_deg)
    w = np.linspace(1.1, 0.8, n).reshape(-1, 1)     # tune here how strongly the weighting 
    rmse_pos = np.sqrt(np.mean(w * err_pos_deg ** 2))
    rmse_vel = np.sqrt(np.mean(w * err_vel_deg ** 2))
    return float(rmse_pos + 0.5 * rmse_vel)


def cost_function_raw(
    params: dict[str, np.ndarray],
    sim_time: float,
    xml_file: Path | None,
) -> float:
    """Set all parameter groups, replay the measured forces, return the fit cost."""
    model, data = get_local_model_and_data(xml_file)
    # Physical validity: nothing negative (stiffness/damping/frictionloss ≥ 0).
    if any(np.any(np.asarray(v) < 0.0) for v in params.values()):
        return 1e6
    global _EVAL_COUNTER
    _EVAL_COUNTER += 1
    try:
        set_params(model, params)
        est_qpos, est_qvel, _, _, _ = simulate_and_sample(
            model, data, sim_time, _GT_QPOS[0], record_dt=_RECORD_DT
        )
    except Exception:
        return 1e6

    global _FIRST_COST_PRINTED
    if not _FIRST_COST_PRINTED:
        print(
            f"[cost_function] _GT_QPOS shape={getattr(_GT_QPOS, 'shape', None)}, "
            f"est_qpos shape={getattr(est_qpos, 'shape', None)}"
        )
        _FIRST_COST_PRINTED = True

    cost = compute_cost(_GT_QPOS, _GT_QVEL, est_qpos, est_qvel)

    if should_preview_simulation(_EVAL_COUNTER, _VIEWER_ENABLED, _VIEWER_INTERVAL):
        try:
            preview_simulation_passive(model, _GT_QPOS[0], sim_time, record_dt=_RECORD_DT, title=f"evaluation {_EVAL_COUNTER}")
        except Exception as exc:
            print(f"[viewer] preview skipped: {exc}")
    return cost


def run_validate(
    seeds: dict[str, np.ndarray], sim_time: float, xml_file: Path | None
) -> dict:
    """Set the seed parameters directly, run one forward sim, return results.

    This is the 'validate' step: no optimization, just check how well the
    identified k/d (and the frictionloss seed) already reproduce the real data.
    """
    model, data = get_local_model_and_data(xml_file)
    set_params(model, seeds)
    est_qpos, est_qvel, sim_f0, sim_f1, sim_t = simulate_and_sample(
        model, data, sim_time, _GT_QPOS[0], record_dt=_RECORD_DT
    )
    cost = compute_cost(_GT_QPOS, _GT_QVEL, est_qpos, est_qvel)
    return {
        "est_qpos": est_qpos,
        "est_qvel": est_qvel,
        "sim_f0": sim_f0,
        "sim_f1": sim_f1,
        "sim_t": sim_t,
        "cost": cost,
    }


def write_cost_function_debug_parquet(
    gt_pos_match: np.ndarray,
    est_pos_match: np.ndarray,
    gt_vel_match: np.ndarray,
    est_vel_match: np.ndarray,
    out_path: Path,
) -> None:
    err_pos_deg_debug = np.rad2deg(gt_pos_match - est_pos_match)
    n_debug = len(err_pos_deg_debug)
    w_debug = np.linspace(1.1, 0.8, n_debug)

    data_dict = {"step": np.arange(n_debug), "weight": w_debug}
    nj = gt_pos_match.shape[1]
    for j in range(nj):
        data_dict[f"gt_pos_deg_j{j}"] = np.rad2deg(gt_pos_match[:, j])
        data_dict[f"est_pos_deg_j{j}"] = np.rad2deg(est_pos_match[:, j])
        data_dict[f"gt_vel_deg_j{j}"] = np.rad2deg(gt_vel_match[:, j])
        data_dict[f"est_vel_deg_j{j}"] = np.rad2deg(est_vel_match[:, j])
        data_dict[f"err_deg_j{j}"] = err_pos_deg_debug[:, j]
        data_dict[f"weighted_sq_err_j{j}"] = w_debug * (err_pos_deg_debug[:, j] ** 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(data_dict).write_parquet(out_path)


def global_objective(
    x: np.ndarray,
    specs: list[GroupSpec],
    seeds: dict[str, np.ndarray],
    sim_time: float,
    xml_file: Path | None = None,
) -> float:
    """Decode the optimizer vector into all-group params and evaluate the cost."""
    p = decode_params(x, specs, seeds)
    return cost_function_raw(p, sim_time, xml_file)


# Context tuple (specs, seeds, sim_time, xml_file) shared with parallel CMA-ES
# workers. Set before the pool is created so forked children inherit it (same
# Linux-fork reliance as the differential_evolution _GT_* globals).
_CMA_CTX = None


def _cma_eval(x: np.ndarray) -> float:
    """Top-level objective for CMA-ES multiprocessing workers."""
    specs, seeds, sim_time, xml_file = _CMA_CTX
    return global_objective(x, specs, seeds, sim_time, xml_file)


def _pool_init_ignore_sigint() -> None:
    """Pool worker initializer: ignore SIGINT so Ctrl+C is handled only by the
    parent. Otherwise every worker also raises KeyboardInterrupt and the pool
    deadlocks on cleanup."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_de(x0, bounds, ctx, workers, maxiter, tol, pop_mult, polish):
    """Differential-evolution backend. Returns (res, history, cost_history, interrupted)."""
    specs, seeds, sim_time, xml_file = ctx
    history = []
    cost_history = []
    gen_counter = 0
    best_so_far = {"x": np.asarray(x0, dtype=float).copy(), "cost": None}

    def cb(xk, convergence=None):
        nonlocal gen_counter
        gen_counter += 1
        history.append(np.asarray(xk).copy())
        cost = global_objective(xk, *ctx)  # 1 extra sim/gen for logging + interrupt readout
        cost_history.append(cost)
        best_so_far["x"] = np.asarray(xk, dtype=float).copy()
        best_so_far["cost"] = cost
        conv = 0.0 if convergence is None else float(convergence)
        print(f"[de gen {gen_counter:03d}/{maxiter:03d}] best_cost={cost:.6f} convergence={conv:.3e}")

    popsize_int = max(1, int(round(pop_mult)))
    total_population = popsize_int * len(bounds)
    if total_population < max(1, workers):
        popsize_int = int(np.ceil(workers / max(1, len(bounds))))
        total_population = popsize_int * len(bounds)
    print(f"[de setup] popsize={popsize_int}/dim, total_population={total_population}, polish={polish}")

    interrupted = False
    try:
        res = differential_evolution(
            global_objective, args=ctx, bounds=bounds, x0=x0,
            maxiter=maxiter, tol=tol, seed=42, polish=polish,
            popsize=popsize_int, workers=workers, disp=False, callback=cb,
        )
    except KeyboardInterrupt:
        interrupted = True
        best_x = best_so_far["x"]
        best_cost = best_so_far["cost"]
        if best_cost is None:
            best_cost = global_objective(best_x, *ctx)
        res = SimpleNamespace(x=best_x, fun=float(best_cost))
        print(f"\n[de] KeyboardInterrupt after {gen_counter} generation(s) — keeping best cost={res.fun:.6f}")
    return res, history, cost_history, interrupted


def _run_cma(x0, bounds, ctx, workers, maxiter, sigma0):
    """CMA-ES backend. Returns (res, history, interrupted).

    The search runs in a normalized [0, 1] box so the very different physical
    scales (stiffness ~0.5, damping ~1e-3, poly coefficients) don't cripple the
    single step-size sigma0. x0 is the seed (mapped to ~0.5, i.e. box centre).
    """
    global _CMA_CTX
    specs, seeds, sim_time, xml_file = ctx
    cost_history = []
    try:
        import cma
    except ImportError as exc:
        raise RuntimeError("CMA-ES needs the 'cma' package — install with: uv add cma") from exc

    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)
    x0n = np.clip((np.asarray(x0, dtype=float) - lo) / span, 0.0, 1.0)

    def denorm(xn):
        return lo + np.asarray(xn, dtype=float) * span

    cma_default = 4 + int(3 * np.log(len(bounds)))
    popsize = max(cma_default, int(workers))  # keep all workers busy
    es = cma.CMAEvolutionStrategy(
        list(x0n), sigma0,
        {"bounds": [0, 1], "maxiter": maxiter, "popsize": popsize, "seed": 42, "verbose": -9},
    )
    print(f"[cma setup] popsize={popsize}, sigma0={sigma0}, maxiter={maxiter}")

    history = []
    gen = 0
    interrupted = False
    best_so_far = {"x": denorm(x0n).copy(), "cost": None}

    pool = None
    if workers and workers > 1:
        import multiprocessing as mp
        _CMA_CTX = ctx                 # set BEFORE fork so children inherit it
        pool = mp.Pool(processes=workers, initializer=_pool_init_ignore_sigint)
    try:
        while not es.stop():
            gen += 1
            sols = es.ask()
            phys = [denorm(s) for s in sols]
            if pool is not None:
                costs = pool.map(_cma_eval, phys)
            else:
                costs = [global_objective(p, *ctx) for p in phys]
            es.tell(sols, costs)
            fbest = float(es.result.fbest)
            xbest = denorm(es.result.xbest)
            history.append(xbest.copy())
            cost_history.append(fbest)
            best_so_far["x"] = xbest.copy()
            best_so_far["cost"] = fbest
            print(f"[cma gen {gen:03d}/{maxiter:03d}] best_cost={fbest:.6f} sigma={es.sigma:.3e}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[cma] KeyboardInterrupt after {gen} generation(s) — keeping best-so-far.")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    best_x = best_so_far["x"]
    best_cost = best_so_far["cost"]
    if best_cost is None:
        best_cost = global_objective(best_x, *ctx)
    res = SimpleNamespace(x=best_x, fun=float(best_cost))
    return res, history, cost_history, interrupted


def optimize_parameters(
    groups: list[ParamGroupConfig],
    seeds: dict[str, np.ndarray],
    sim_time: float,
    xml_file: Path | None,
    optimizer: str = "de",
    workers: int = 1,
    maxiter: int = 10,
    tol: float = 0.01,
    pop_mult: float = 5.0,
    polish: bool = True,
    sigma0: float = 0.25,
):
    """Finetune the active groups with the selected optimizer ('de' or 'cma').

    Only groups with ``optimize=True`` contribute variables; their
    representation ('per_joint' or 'poly') sets how many. The search is seeded
    (x0) with the identified values and bounded to a relative band around them.

    A KeyboardInterrupt (Ctrl+C) stops cleanly and keeps the best solution so
    far. Returns (res, best, specs, history, interrupted).
    """
    specs, x0, bounds = build_param_space(groups, seeds)
    if not bounds:
        raise RuntimeError("No parameter groups selected for optimization")

    ctx = (specs, seeds, sim_time, xml_file)
    active = ", ".join(f"{s.name}[{s.representation}:{s.slice[1] - s.slice[0]}]" for s in specs)
    print(f"[optimizer={optimizer}] active={active} | nvars={len(bounds)} | workers={workers}")
    print("[hint] press Ctrl+C to stop early and keep the best result so far.")

    if optimizer == "de":
        res, history, cost_history, interrupted = _run_de(x0, bounds, ctx, workers, maxiter, tol, pop_mult, polish)
    elif optimizer == "cma":
        res, history, cost_history, interrupted = _run_cma(x0, bounds, ctx, workers, maxiter, sigma0)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}' (choose 'de' or 'cma')")

    best = decode_params(res.x, specs, seeds)
    return res, best, specs, history, cost_history, interrupted


def save_validation_plot(
    gt_qpos: np.ndarray,
    est_qpos: np.ndarray,
    out_path: Path,
    gt_force_times: np.ndarray | None = None,
    gt_f0: np.ndarray | None = None,
    gt_f1: np.ndarray | None = None,
    sim_times: np.ndarray | None = None,
    sim_f0: np.ndarray | None = None,
    sim_f1: np.ndarray | None = None,
    joint_labels: list[dict] | None = None,
):
    if plt is None:
        return
    njnt = gt_qpos.shape[1]

    def _line_label(i: int) -> str:
        # column i is model joint index i (canonical order: 0 = base … njnt-1 = tip)
        end = "  ·base" if i == 0 else ("  ·tip" if i == njnt - 1 else "")
        if joint_labels and i < len(joint_labels):
            lab = joint_labels[i]
            return f"idx {i:>2}: {lab['model_joint']} = joint_{lab['real_joint']}{end}"
        return f"model idx {i}{end}"

    # Distinct colour per joint (tab20 gives 20 well-separated hues).
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    colors = plt.cm.tab20(np.linspace(0, 1, max(njnt, 2)))

    # Same colour = same joint in BOTH trajectory subplots.
    for i in range(njnt):
        axs[0].plot(gt_qpos[:, i], color=colors[i], linewidth=1.3, label=_line_label(i))
    axs[0].set_title(
        "Ground-truth joint angles  —  line colour = joint "
        "(model index 0 = base / j_%d … %d = tip / j_0)" % (njnt - 1, njnt - 1)
    )
    axs[0].set_ylabel("angle (rad)")

    for i in range(njnt):
        axs[1].plot(est_qpos[:, i], color=colors[i], linewidth=1.3, label=_line_label(i))
    axs[1].set_title("Simulated joint angles (identified params) — same colours as above")
    axs[1].set_ylabel("angle (rad)")
    # one shared legend (right of the top subplot) mapping colour → joint
    axs[0].legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8,
                  title="colour → joint\n(model idx: j_ = real)", title_fontsize=8)

    # Forces: plot GT forces (if provided) and applied sim forces (if provided)
    if (gt_force_times is not None and gt_f0 is not None) or (sim_times is not None and sim_f0 is not None):
        if gt_force_times is not None and gt_f0 is not None:
            axs[2].plot(gt_force_times, gt_f0, color="steelblue", label="gt force 0")
        if gt_force_times is not None and gt_f1 is not None:
            axs[2].plot(gt_force_times, gt_f1, color="darkorange", label="gt force 1")
        if sim_times is not None and sim_f0 is not None:
            axs[2].plot(sim_times, sim_f0, color="navy", linestyle="--", label="sim applied f0")
        if sim_times is not None and sim_f1 is not None:
            axs[2].plot(sim_times, sim_f1, color="orangered", linestyle="--", label="sim applied f1")
        axs[2].set_title("Forces: ground-truth (solid) vs applied (dashed)")
        axs[2].set_ylabel("force (N)")
        axs[2].legend(loc='best')

    axs[-1].set_xlabel("time (samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")  # keep the outside legend
    plt.close(fig)


def save_cost_history_plot(
    cost_history: list[float], out_path: Path, optimizer: str = "", baseline: float | None = None
) -> None:
    """Plot the best-so-far cost against optimizer iteration (generation)."""
    if plt is None or not cost_history:
        return
    gens = np.arange(1, len(cost_history) + 1)
    ch = np.asarray(cost_history, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, ch, marker="o", ms=4, color="#2b8cbe", label="best cost so far")
    if baseline is not None:
        ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"validate baseline ({baseline:.3f})")
    ax.scatter([gens[-1]], [ch[-1]], color="#e34a33", zorder=5,
               label=f"final ({ch[-1]:.3f})")
    ax.set_title(f"Convergence — {optimizer.upper()} best cost vs iteration")
    ax.set_xlabel("iteration (generation)")
    ax.set_ylabel("cost")
    # log-y only helps when the range spans orders of magnitude and stays positive
    if ch.min() > 0 and ch.max() / max(ch.min(), 1e-9) > 50:
        ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_params_over_joints_plot(
    seeds: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    joint_labels: list[dict],
    anchors: dict[int, dict],
    out_path: Path,
    values_label: str = "identified",
) -> None:
    """One subplot per joint-domain group (stiffness/damping/frictionloss): the
    per-joint value over the model joint index, seed vs identified/optimized,
    with the measured anchor joints marked."""
    if plt is None:
        return
    joint_groups = [g for g in GROUP_ORDER if GROUP_DOMAIN[g] == "joint"]
    if not joint_groups:
        return

    njnt = len(seeds[joint_groups[0]])
    idx = np.arange(njnt)
    # x tick labels: "idx\nj_x\njoint_y"
    xticklabels = [
        f"{lab['model_index']}\n{lab['model_joint']}\njoint_{lab['real_joint']}"
        for lab in joint_labels
    ]
    # anchor model indices (real joint number → model index), for k/d overlays
    real_to_model = {lab["real_joint"]: lab["model_index"] for lab in joint_labels}
    anchor_idx = sorted(
        {real_to_model[min(int(n), njnt)] for n in anchors if min(int(n), njnt) in
         set(real_to_model)}
    )

    fig, axs = plt.subplots(len(joint_groups), 1, figsize=(13, 3.4 * len(joint_groups)), sharex=True)
    if len(joint_groups) == 1:
        axs = [axs]
    ylabels = {"stiffness": "stiffness [Nm/rad]", "damping": "damping [Nm·s/rad]",
               "frictionloss": "frictionloss"}
    for ax, g in zip(axs, joint_groups):
        s = np.asarray(seeds[g], dtype=float)
        v = np.asarray(values[g], dtype=float)
        ax.plot(idx, s, ls="--", color="gray", marker="o", ms=3, label="seed (identified)")
        if not np.allclose(s, v):
            ax.plot(idx, v, ls="-", color="#238b45", marker="o", ms=4, label=values_label)
        # mark measured anchor joints ON the seed line (only for stiffness/damping)
        if g in ("stiffness", "damping") and anchor_idx:
            ax.scatter(anchor_idx, s[anchor_idx], facecolors="none", edgecolors="#e34a33",
                       s=90, lw=1.8, zorder=5, label="measured anchor")
        # log-y when the values span orders of magnitude (seed tiny vs optimized large)
        both = np.concatenate([s[s > 0], v[v > 0]])
        if both.size and both.max() / both.min() > 30:
            ax.set_yscale("log")
        ax.set_ylabel(ylabels.get(g, g))
        ax.set_title(g)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
    axs[-1].set_xticks(idx)
    axs[-1].set_xticklabels(xticklabels, fontsize=7)
    axs[-1].set_xlabel("model joint index  /  model joint  /  real joint   (0 = base … tip)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_groups(
    base_groups: list[ParamGroupConfig],
    optimize_csv: str | None,
    represent_csv: str | None,
    band_override: float | None,
    bounds_csv: str | None = None,
) -> list[ParamGroupConfig]:
    """Apply CLI overrides (--optimize / --represent / --band / --bounds) onto the config."""
    groups = [replace(g) for g in base_groups]
    by_name = {g.name: g for g in groups}

    if optimize_csv is not None:
        want = {t.strip() for t in optimize_csv.split(",") if t.strip()}
        unknown = want - set(by_name)
        if unknown:
            raise SystemExit(f"--optimize: unknown group(s) {sorted(unknown)}; valid: {list(by_name)}")
        for g in groups:
            g.optimize = g.name in want

    if represent_csv:
        for tok in represent_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise SystemExit(f"--represent: expected name=repr[:degree], got '{tok}'")
            name, rep = (s.strip() for s in tok.split("=", 1))
            if name not in by_name:
                raise SystemExit(f"--represent: unknown group '{name}'; valid: {list(by_name)}")
            deg = None
            if ":" in rep:
                rep, ds = rep.split(":", 1)
                deg = int(ds)
            if rep not in ("per_joint", "shared", "poly"):
                raise SystemExit(f"--represent: unknown representation '{rep}' (per_joint|shared|poly)")
            by_name[name].representation = rep
            if deg is not None:
                by_name[name].poly_degree = deg

    if band_override is not None:
        for g in groups:
            g.band = band_override

    # --bounds "damping=1e-4:0.1,stiffness=0.1:2.0" → absolute range (overrides band)
    if bounds_csv:
        for tok in bounds_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok or ":" not in tok.split("=", 1)[1]:
                raise SystemExit(f"--bounds: expected name=min:max, got '{tok}'")
            name, rng = (s.strip() for s in tok.split("=", 1))
            if name not in by_name:
                raise SystemExit(f"--bounds: unknown group '{name}'; valid: {list(by_name)}")
            lo_s, hi_s = rng.split(":", 1)
            lo, hi = float(lo_s), float(hi_s)
            if not hi > lo:
                raise SystemExit(f"--bounds: min must be < max for '{name}' (got {lo}:{hi})")
            by_name[name].bounds = (lo, hi)

    return groups


def print_seed_table(
    seeds: dict[str, np.ndarray],
    anchors: dict[int, dict],
    used: list[int],
    joint_labels: list[dict],
) -> None:
    njnt = len(seeds["stiffness"])
    print(f"\n  Identified anchors from data/free_vibration (used real joints: {used}):")
    for n in sorted(anchors):
        a = anchors[n]
        if n not in used:
            flag = "  (shadowed by another anchor → unused)"
        elif n > njnt:
            flag = f"  (→ clamped onto last joint joint_{njnt})"
        else:
            flag = ""
        print(f"    joint_{n:>2}: k={a['k']:.5f} Nm/rad, d={a['d']:.6f} Nm·s/rad{flag}")
    print("\n  Per-joint seeds — arrays are ordered by MODEL joint index:")
    print(f"    {'idx':>3} {'model':>6} {'real':>9} {'stiffness':>12} {'damping':>12} {'frictionloss':>13}")
    for lab in joint_labels:
        i = lab["model_index"]
        print(
            f"    {i:>3} {lab['model_joint']:>6} {'joint_' + str(lab['real_joint']):>9} "
            f"{seeds['stiffness'][i]:>12.5f} {seeds['damping'][i]:>12.6f} "
            f"{seeds['frictionloss'][i]:>13.5f}"
        )
    # tendon seeds (read from the model/XML)
    ntendon = len(seeds["tendon_stiffness"])
    print("\n  Per-tendon seeds (from XML):")
    print(f"    {'tendon':>6} {'stiffness':>12} {'damping':>12} {'frictionloss':>13}")
    for t in range(ntendon):
        print(
            f"    {t:>6} {seeds['tendon_stiffness'][t]:>12.5f} "
            f"{seeds['tendon_damping'][t]:>12.5f} {seeds['tendon_frictionloss'][t]:>13.5f}"
        )
    # solver-option seeds (from XML)
    opt_names = [n for n, d in GROUP_DOMAIN.items() if d == "opt"]
    if opt_names:
        print("\n  Solver option seeds (model.opt, from XML):")
        for name in opt_names:
            print(f"    {name} = {float(seeds[name][0]):.5f}")
    # broadcast solver-knob seeds (from XML)
    if BROADCAST_KNOBS:
        print("\n  Solver knob seeds (broadcast to all joints/dofs, from XML):")
        for name, (attr, col, _bnd) in BROADCAST_KNOBS.items():
            loc = attr if col is None else f"{attr}[:, {col}]"
            print(f"    {name:<22} = {float(seeds[name][0]):>10.5f}   ({loc})")


def _gt_forces() -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Ground-truth force curves sampled at the recorded timestamps (for plots)."""
    try:
        if _SIM_TIMESTEPS is None:
            return None, None, None
        gt_f0 = _FORCE_INTERP_0(_SIM_TIMESTEPS) if _FORCE_INTERP_0 is not None else None
        gt_f1 = _FORCE_INTERP_1(_SIM_TIMESTEPS) if _FORCE_INTERP_1 is not None else None
        return _SIM_TIMESTEPS, gt_f0, gt_f1
    except Exception:
        return None, None, None


def write_params_json(
    path: Path,
    params: dict[str, np.ndarray],
    cost: float,
    mode: str,
    groups: list[ParamGroupConfig],
    anchors: dict[int, dict],
    used: list[int],
    joint_labels: list[dict],
) -> None:
    st = np.asarray(params["stiffness"]).tolist()
    dp = np.asarray(params["damping"]).tolist()
    fl = np.asarray(params["frictionloss"]).tolist()

    # Explicit, unambiguous per-joint records (index i = model joint i).
    joints = [
        {
            "model_index": lab["model_index"],
            "model_joint": lab["model_joint"],        # e.g. "j_10"
            "real_joint": f"joint_{lab['real_joint']}",  # e.g. "joint_11"
            "stiffness": st[lab["model_index"]],
            "damping": dp[lab["model_index"]],
            "frictionloss": fl[lab["model_index"]],
        }
        for lab in joint_labels
    ]

    out = {
        "mode": mode,
        "cost": float(cost),
        # How to read every per-joint array below.
        "joint_order": (
            "All per-joint arrays are ordered by MuJoCo MODEL joint index i "
            "(0..njnt-1). index i = model joint 'model_joint' = real 'real_joint'. "
            "The model index follows the XML kinematic tree from the base: index 0 "
            "= bottom/base joint (j_%d), index njnt-1 = tip (j_0). 'joints' lists the "
            "explicit per-joint mapping; the flat arrays match it element-wise and "
            "index directly into model.jnt_stiffness / model.dof_damping / "
            "model.dof_frictionloss."
        ) % (len(joint_labels) - 1),
        "joints": joints,
        "stiffness": st,
        "damping": dp,
        "frictionloss": fl,
        # tendon params (one value per tendon, ordered by MuJoCo tendon index)
        "tendons": [
            {
                "tendon_index": t,
                "tendon_stiffness": float(params["tendon_stiffness"][t]),
                "tendon_damping": float(params["tendon_damping"][t]),
                "tendon_frictionloss": float(params["tendon_frictionloss"][t]),
            }
            for t in range(len(params["tendon_stiffness"]))
        ],
        "tendon_stiffness": np.asarray(params["tendon_stiffness"]).tolist(),
        "tendon_damping": np.asarray(params["tendon_damping"]).tolist(),
        "tendon_frictionloss": np.asarray(params["tendon_frictionloss"]).tolist(),
        # solver options (scalars written to model.opt.<name>)
        "options": {
            name: float(np.asarray(params[name]).ravel()[0])
            for name, domain in GROUP_DOMAIN.items()
            if domain == "opt" and name in params
        },
        # broadcast solver knobs (scalars written across all joints/dofs)
        "solver_knobs": {
            name: float(np.asarray(params[name]).ravel()[0])
            for name in BROADCAST_KNOBS
            if name in params
        },
        "identified_anchors": {f"joint_{n}": anchors[n] for n in sorted(anchors)},
        "used_anchor_joints": [f"joint_{n}" for n in used],
        "groups": [
            {
                "name": g.name,
                "optimize": g.optimize,
                "representation": g.representation,
                "poly_degree": g.poly_degree,
                "band": g.band,
                "bounds": list(g.bounds) if g.bounds is not None else None,
            }
            for g in groups
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved params to: {path}")


def save_identified_xml(
    params: dict[str, np.ndarray], xml_file: Path | None, out_path: Path
) -> None:
    """Write a MuJoCo XML with the identified/optimized parameters baked in.

    Loads the base model (which sets MuJoCo's 'last-loaded XML' buffer), applies
    all groups via set_params (per-joint stiffness/damping/frictionloss, tendon
    params, broadcast solver knobs like solref/solimp/armature, and opt scalars
    like impratio), then serializes with mj_saveLastXML. The result is a
    self-contained, reloadable model.
    """
    model = make_model(xml_file)   # (re)sets the 'last loaded' XML buffer
    set_params(model, params)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mj.mj_saveLastXML(str(out_path), model)
    except Exception as exc:
        print(f"[xml] could not save identified XML: {exc}")
        return
    print(f"Saved identified model XML to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-time", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--inner-maxiter", type=int, default=4)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--xml", type=str, default=None, help="Model XML path (default: models/spirob_13seg.xml)")
    ap.add_argument("--outlier-thresh", type=float, default=3.5, help="MAD z-score threshold for outlier detection")
    ap.add_argument("--simple-outlier", action="store_true", help="Use simple neighbor-diff outlier detection")
    ap.add_argument("--outlier-abs-thresh", type=float, default=0.1, help="Absolute threshold for simple outlier detection (units: same as signal)")
    ap.add_argument("--record-dt", type=float, default=0.0, help="Recording interval in seconds (e.g., 0.1 for 10 Hz); 0 = no downsampling")
    ap.add_argument(
        "--reverse-real-joints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reverse real joint order to match simulation indexing (default: module global)",
    )
    ap.add_argument("--preprocess-only", action="store_true", help="Run only preprocessing and save plots, skip everything else")
    ap.add_argument("--tol", type=float, default=0.01, help="Tolerance for differential_evolution convergence")
    ap.add_argument("--pop-mult", type=float, default=5.0, help="Multiplier for DE popsize (per-dimension). total_pop = pop_mult * nvars")
    ap.add_argument("--settling-steps", type=int, default=DEFAULT_SETTLING_STEPS, help="Quasi-static settling steps before each replay (default 1000; lower = faster, ~65%% of per-eval cost)")
    ap.add_argument("--no-polish", action="store_true", help="Skip the DE polish (L-BFGS-B) step — much faster, slightly less refined")
    ap.add_argument("--enable-viewer", action=argparse.BooleanOptionalAction, default=False, help="Show passive MuJoCo previews during optimization")
    ap.add_argument("--viewer-interval", type=int, default=100, help="Show the first evaluation and then every Nth evaluation in the passive viewer")
    # ── Modular parameter identification ──
    ap.add_argument("--mode", choices=["validate", "finetune"], default="validate",
                    help="validate: set identified seeds and run one forward sim. finetune: additionally optimize.")
    ap.add_argument("--optimize", type=str, default=None,
                    help="Comma list of groups to optimize (subset of stiffness,damping,frictionloss). Overrides the config block.")
    ap.add_argument("--represent", type=str, default=None,
                    help="Per-group representation, e.g. 'stiffness=per_joint,frictionloss=poly:2'.")
    ap.add_argument("--band", type=float, default=None, help="Global relative search band override for all groups (e.g. 0.3).")
    ap.add_argument("--bounds", type=str, default=None, help="Absolute per-group ranges (overrides band), e.g. 'damping=1e-4:0.1,stiffness=0.1:2.0'.")
    ap.add_argument("--frictionloss-init", type=float, default=FRICTIONLOSS_INIT, help="Uniform seed for frictionloss (no measurement).")
    ap.add_argument("--optimizer", choices=["de", "cma"], default="de",
                    help="Search algorithm: 'de' (differential evolution) or 'cma' (CMA-ES).")
    ap.add_argument("--cma-sigma0", type=float, default=0.25,
                    help="CMA-ES initial step size in the normalized [0,1] search box (0.2-0.3 typical).")
    args = ap.parse_args()

    if args.data:
        data_path = Path(args.data)
    else:
        base = DEFAULT_TRAJECTORY.parent
        cand1 = DEFAULT_TRAJECTORY
        cand2 = base / "sys_id_auto.parquet"
        if cand1.exists():
            data_path = cand1
        elif cand2.exists():
            data_path = cand2
        else:
            print(f"Data file not found in {base} (tried {cand1.name} and {cand2.name})")
            return

    xml_file = Path(args.xml) if args.xml else None

    print("Loading and preprocessing data...")
    global _RECORD_DT, _REVERSE_REAL_JOINT_ORDER, _ACTUAL_SIM_TIME
    _RECORD_DT = float(args.record_dt)
    if args.reverse_real_joints is not None:
        _REVERSE_REAL_JOINT_ORDER = bool(args.reverse_real_joints)
    global _VIEWER_ENABLED, _VIEWER_INTERVAL, _SETTLING_STEPS
    _VIEWER_ENABLED = bool(args.enable_viewer)
    _VIEWER_INTERVAL = max(1, int(args.viewer_interval))
    _SETTLING_STEPS = max(0, int(args.settling_steps))

    meta = load_and_preprocess(
        data_path,
        args.sim_time,
        outlier_thresh=args.outlier_thresh,
        simple_outlier=args.simple_outlier,
        outlier_abs_thresh=args.outlier_abs_thresh,
        record_dt=_RECORD_DT,
    )

    real_samples = len(meta["t"]) if meta and "t" in meta else 0
    real_duration = meta["t"][-1] if meta and "t" in meta and len(meta["t"]) > 0 else 0
    print(f"Preprocessing: real samples after downsampling: {real_samples}, real_duration: {real_duration:.4f}s")
    print(f"Requested sim_time: {args.sim_time:.4f}s, actual_sim_time: {_ACTUAL_SIM_TIME:.4f}s")

    if args.preprocess_only:
        print("Preprocessing complete — saved plots to build/preprocessing_plots. Exiting due to --preprocess-only.")
        return

    # ── Build seeds (joint groups from identified anchors, tendon groups from XML) ──
    seed_model = make_model(xml_file)
    seeds, anchors, used = build_seeds(seed_model, _REVERSE_REAL_JOINT_ORDER, args.frictionloss_init)
    # Canonical index → (model joint name, real joint number) mapping used for
    # every human-readable output (seed table, plots, JSON).
    joint_labels = joint_index_labels(seed_model, _REVERSE_REAL_JOINT_ORDER)
    print_seed_table(seeds, anchors, used, joint_labels)

    groups = resolve_groups(DEFAULT_PARAM_GROUPS, args.optimize, args.represent, args.band, args.bounds)
    build_dir = _build_dir("real2sim")
    gt_ft, gt_f0, gt_f1 = _gt_forces()

    # ── Step 1: VALIDATE — how well do the identified seeds already fit? ──
    print("\n=== VALIDATE: forward simulation with identified seeds ===")
    val = run_validate(seeds, _ACTUAL_SIM_TIME, xml_file)
    print(f"[validate] cost = {val['cost']:.6f}")
    save_validation_plot(
        _GT_QPOS, val["est_qpos"], build_dir / "validate.png",
        gt_force_times=gt_ft, gt_f0=gt_f0, gt_f1=gt_f1,
        sim_times=val["sim_t"], sim_f0=val["sim_f0"], sim_f1=val["sim_f1"],
        joint_labels=joint_labels,
    )
    print(f"Saved validation plot to: {build_dir / 'validate.png'}")
    save_params_over_joints_plot(
        seeds, seeds, joint_labels, anchors,
        build_dir / "validate_params.png", values_label="identified (seed)",
    )
    write_params_json(build_dir / "validate.json", seeds, val["cost"],
                      "validate", groups, anchors, used, joint_labels)

    if args.mode == "validate":
        print("\nMode 'validate' → done (no optimization). Use --mode finetune to optimize.")
        return

    # ── Step 2: FINETUNE — optimize the selected groups around the seeds ──
    active = [g.name for g in groups if g.optimize]
    if not active:
        print("\nNo groups selected for optimization (--optimize/config). Nothing to finetune.")
        return
    print(f"\n=== FINETUNE: optimizing {active} around identified seeds ===")
    if _VIEWER_ENABLED and args.workers != 1:
        print("[viewer] enabled with workers != 1; previews may be skipped. Use --workers 1 for reliable preview.")

    res, best, specs, history, cost_history, interrupted = optimize_parameters(
        groups, seeds, args.sim_time, xml_file,
        optimizer=args.optimizer,
        workers=args.workers, maxiter=args.inner_maxiter, tol=args.tol, pop_mult=args.pop_mult,
        polish=not args.no_polish, sigma0=args.cma_sigma0,
    )
    tag = " (interrupted)" if interrupted else ""
    print(f"[finetune] best cost = {float(res.fun):.6f}{tag}  (validate was {val['cost']:.6f})")

    model, data = get_local_model_and_data(xml_file)
    set_params(model, best)
    est_qpos, _, sim_f0, sim_f1, sim_t = simulate_and_sample(
        model, data, _ACTUAL_SIM_TIME, _GT_QPOS[0], record_dt=_RECORD_DT,
    )
    save_validation_plot(
        _GT_QPOS, est_qpos, build_dir / "finetune.png",
        gt_force_times=gt_ft, gt_f0=gt_f0, gt_f1=gt_f1,
        sim_times=sim_t, sim_f0=sim_f0, sim_f1=sim_f1,
        joint_labels=joint_labels,
    )
    print(f"Saved finetune plot to: {build_dir / 'finetune.png'}")
    # convergence plot: cost over iterations
    save_cost_history_plot(
        cost_history, build_dir / "finetune_cost.png",
        optimizer=args.optimizer, baseline=val["cost"],
    )
    # per-joint parameter distribution: seed vs optimized
    save_params_over_joints_plot(
        seeds, best, joint_labels, anchors,
        build_dir / "finetune_params.png", values_label="optimized",
    )
    print("Saved cost-history + parameter plots to build/")
    write_params_json(build_dir / "finetune.json", best, float(res.fun),
                      "finetune", groups, anchors, used, joint_labels)
    # write a ready-to-use MuJoCo XML with the optimized parameters baked in
    save_identified_xml(best, xml_file, build_dir / "finetune.xml")


if __name__ == "__main__":
    main()
