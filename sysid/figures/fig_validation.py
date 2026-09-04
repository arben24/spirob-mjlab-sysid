#!/usr/bin/env python3
"""Publication-quality validation figure of the real-to-sim identification.

Four panels over one shared time axis:

    a  measured joint angles (ArUco tracking)          -- ground truth
    b  simulated joint angles with the identified parameters
    c  residual per joint (simulation - measurement)
    d  the measured tendon forces that drive the simulation

The parameters come from the JSON of a ``real2sim.py`` run (default: the
CMA-ES fine-tune). Panel b is re-simulated here -- same preprocessing, same
model, same force trace as in the optimization run -- so the figure shows
exactly what the cost function scored.

Colour: the 13 joints are an *ordered* quantity (base -> tip), so they get one
sequential ramp with a colour bar rather than 13 categorical colours. Panel c
has two series and uses the categorical slots from ``spirob.plotstyle``.

Usage::

    uv run sysid/figures/fig_validation.py                        # CMA-ES
    uv run sysid/figures/fig_validation.py \\
        --params data/identified/real2sim_de_500iter.json \\
        --label "Differential Evolution" --out build/figures/fig_validation_de

Inputs : data/identified/*.json, data/trajectories/*.parquet, models/*.xml
Outputs: build/figures/fig_real2sim_validation.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation_based"))
import real2sim as srm  # noqa: E402

from spirob import plotstyle as ts  # noqa: E402
from spirob.paths import DEFAULT_PARAMS, DEFAULT_TRAJECTORY
from spirob.paths import build_dir as _build_dir

BUILD = _build_dir("figures")

# Single-hue sequential ramp (light -> dark) for the ordered joint index.
# Lower bound 0.40 so even the lightest joint still reads on white.
CMAP = plt.cm.Blues
RAMP_LO, RAMP_HI = 0.40, 0.98


def joint_colors(njnt: int) -> np.ndarray:
    return CMAP(np.linspace(RAMP_LO, RAMP_HI, njnt))


def load_params(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Build the ``params`` dict for ``srm.set_params`` from a result JSON."""
    raw = json.loads(path.read_text())
    params: dict[str, np.ndarray] = {}
    for key in ("stiffness", "damping", "frictionloss",
                "tendon_stiffness", "tendon_damping", "tendon_frictionloss"):
        if key in raw:
            params[key] = np.asarray(raw[key], dtype=float)
    for key, val in (raw.get("options") or {}).items():
        params[key] = np.asarray([float(val)])
    for key, val in (raw.get("solver_knobs") or {}).items():
        params[key] = np.asarray([float(val)])
    return params, raw


def simulate(params: dict[str, np.ndarray], xml_file: Path | None):
    model, data = srm.get_local_model_and_data(xml_file)
    srm.set_params(model, params)
    est_qpos, _qvel, sim_f0, sim_f1, sim_t = srm.simulate_and_sample(
        model, data, srm._ACTUAL_SIM_TIME, srm._GT_QPOS[0], record_dt=srm._RECORD_DT,
    )
    return est_qpos, sim_t, sim_f0, sim_f1


def residuals(gt_t, gt_qpos, sim_t, est_qpos) -> np.ndarray:
    """Residual (simulation - measurement) in degrees on the measurement time grid.

    The simulation runs at a fixed time step and produces far more
    samples than the tracking, so it is interpolated per joint onto ``gt_t``
    interpolated. Every residual therefore carries a real timestamp, unlike
    the pure length matching in ``srm.resample_match``.
    """
    est_on_gt = np.column_stack([
        np.interp(gt_t, sim_t, est_qpos[:, i]) for i in range(gt_qpos.shape[1])
    ])
    return np.rad2deg(est_on_gt - gt_qpos)


def rmse_per_joint(resid_deg: np.ndarray) -> np.ndarray:
    """Per-joint RMSE in degrees."""
    return np.sqrt(np.mean(resid_deg ** 2, axis=0))


def build_figure(meta, est_qpos, sim_t, label, cost, out_stem, ghost=True):
    gt_qpos = meta["qpos"]
    gt_t = meta["t"]
    njnt = gt_qpos.shape[1]
    colors = joint_colors(njnt)

    resid = residuals(gt_t, gt_qpos, sim_t, est_qpos)
    rmse = rmse_per_joint(resid)

    ts.apply_style()
    fig, axs = plt.subplots(
        4, 1, figsize=(ts.FIG_FULL, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.72, 0.62], "hspace": 0.04},
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.10, hspace=0.0)
    ax_gt, ax_sim, ax_res, ax_f = axs

    # ── a) Ground Truth ────────────────────────────────────────────────
    for i in range(njnt):
        ax_gt.plot(gt_t, np.rad2deg(gt_qpos[:, i]), color=colors[i], linewidth=1.1)
    ax_gt.set_ylabel("Joint angle (°)")
    ax_gt.set_title("a   Measurement — joint angles from ArUco tracking",
                    loc="left", pad=6)

    # ── b) Simulation ──────────────────────────────────────────────────
    if ghost:
        for i in range(njnt):
            ax_sim.plot(gt_t, np.rad2deg(gt_qpos[:, i]),
                        color=ts.FAINT, linewidth=0.7, zorder=1)
    for i in range(njnt):
        ax_sim.plot(sim_t, np.rad2deg(est_qpos[:, i]),
                    color=colors[i], linewidth=1.1, zorder=2)
    ax_sim.set_ylabel("Joint angle (°)")
    ax_sim.set_title(f"b   Simulation — identified parameters ({label})",
                     loc="left", pad=6)
    if ghost:
        ax_sim.text(0.995, 1.02, "grey: measurement from a", transform=ax_sim.transAxes,
                    ha="right", va="bottom", fontsize=8, color=ts.MUTED)

    # Shared y-scale — only then are the two panels directly comparable.
    lo = min(ax_gt.get_ylim()[0], ax_sim.get_ylim()[0])
    hi = max(ax_gt.get_ylim()[1], ax_sim.get_ylim()[1])
    ax_gt.set_ylim(lo, hi)
    ax_sim.set_ylim(lo, hi)

    # ── c) Residuum ────────────────────────────────────────────────────
    # Zero line first so it sits below the curves; symmetric scale,
    # so over- and under-estimation carry equal visual weight.
    ax_res.axhline(0.0, color=ts.BASELINE, linewidth=0.9, zorder=1)
    for i in range(njnt):
        ax_res.plot(gt_t, resid[:, i], color=colors[i], linewidth=1.0, zorder=2)
    span = float(np.abs(resid).max())
    # Strip of space at the bottom for the metrics block
    ax_res.set_ylim(-span * 1.36, span * 1.06)
    ax_res.set_ylabel("Residual (°)")
    ax_res.set_title("c   Residual — simulation minus measurement, per joint",
                     loc="left", pad=6)

    note = (f"cost J = {ts.auto(cost, 4)}   ·   RMSE "
            f"{ts.num(float(np.mean(rmse)), 1)}° mean, "
            f"{ts.num(float(rmse.min()), 1)}°–{ts.num(float(rmse.max()), 1)}° per joint")
    ts.annotate(ax_res, note, loc="lower left")

    # ── d) Anregung ────────────────────────────────────────────────────
    ax_f.plot(gt_t, meta["f0"], label="Tendon 0", **ts.line_kw(0, width=1.3))
    ax_f.plot(gt_t, meta["f1"], label="Tendon 1", **ts.line_kw(1, width=1.3))
    ax_f.set_ylabel("Tendon force (N)")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_title("d   Excitation — measured tendon forces (input to the simulation)",
                   loc="left", pad=6)
    ax_f.legend(loc="upper left", ncol=2)

    for ax in axs:
        ts.grid_on(ax)
    ax_f.set_xlim(gt_t[0], gt_t[-1])

    # ── Colour bar: joint index base -> tip (applies to a, b and c) ────
    sm = ScalarMappable(norm=Normalize(1, njnt),
                        cmap=CMAP.from_list("j", colors))
    cb = fig.colorbar(sm, ax=[ax_gt, ax_sim, ax_res], pad=0.015, aspect=40, fraction=0.035)
    cb.ax.set_title("Joint", fontsize=8, color=ts.INK_2, pad=6)
    cb.set_ticks([1, njnt])
    cb.set_ticklabels(["1 base", f"{njnt} tip"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelcolor=ts.INK_2)

    ts.localize_axes(fig, skip=[cb.ax])
    return fig, rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(DEFAULT_PARAMS),
                    help="Result JSON of a real2sim.py run")
    ap.add_argument("--data", default=None, help="Measurement file (Parquet)")
    ap.add_argument("--xml", default=None, help="Model XML (default: same as the sys-id run)")
    ap.add_argument("--sim-time", type=float, default=60.0)
    ap.add_argument("--outlier-thresh", type=float, default=3.5,
                    help="MAD-z threshold of the outlier rejection — must match the sys-id run")
    ap.add_argument("--label", default="CMA-ES", help="Optimizer name for the panel title")
    ap.add_argument("--out", default=str(BUILD / "fig_real2sim_validation"),
                    help="Output path without extension (.pdf and .png are written)")
    ap.add_argument("--no-ghost", action="store_true",
                    help="Do not draw the measurement faintly behind the simulation")
    args = ap.parse_args()

    data_path = Path(args.data) if args.data else DEFAULT_TRAJECTORY
    if not data_path.exists():
        raise SystemExit(f"Measurement file not found: {data_path}")
    params_path = Path(args.params)
    if not params_path.exists():
        raise SystemExit(f"Parameter JSON not found: {params_path}")

    print(f"Measurements: {data_path}")
    print(f"Parameters:   {params_path}")
    # Important: the same preprocessing as in the optimisation run. The
    # The function default of load_and_preprocess (0.0005) is far sharper than
    # real2sim's CLI default (3.5) and would iron the measurement flat, so the
    # threshold is passed explicitly.
    meta = srm.load_and_preprocess(data_path, args.sim_time,
                                   outlier_thresh=args.outlier_thresh)
    params, raw = load_params(params_path)
    cost = float(raw.get("cost", float("nan")))

    xml_file = Path(args.xml) if args.xml else None
    est_qpos, sim_t, _f0, _f1 = simulate(params, xml_file)
    print(f"Simulated: {est_qpos.shape[0]} samples over {sim_t[-1]:.2f}s")

    fig, rmse = build_figure(meta, est_qpos, sim_t, args.label, cost, args.out,
                             ghost=not args.no_ghost)
    ts.save(fig, args.out)
    print("RMSE per joint (°): " + ", ".join(f"{v:.2f}" for v in rmse))


if __name__ == "__main__":
    main()
