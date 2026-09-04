#!/usr/bin/env python3
"""Torsional stiffness from the static load test.

A Franka Emika Panda holds one segment of a joint and presses the neighbouring
segment onto a precision scale at a known lever arm ``r``, perpendicular to the
lever so ``alpha = 90 deg`` and ``M = F * r`` holds without measuring alpha. The
deflection angle comes from the arm's forward kinematics.

For every measurement point the script computes

    F = m * g            (m from the scale, in grams)
    M = F * r            (r from data/static_load/lever_arms.csv)

and regresses ``M(phi) = k * phi + M0``. The slope is the torsional stiffness.

Two fits are reported. The free-intercept fit (``scipy.stats.linregress``) is
the headline number; its intercept comes out slightly non-zero although
physically it should vanish at phi = 0, so it is printed but kept out of the
figure. The through-origin fit (``k = sum(phi*M) / sum(phi^2)``, M0 = 0 by
construction) is printed alongside — for joint 1 it gives 0.4921 instead of
0.5108 N·m/rad at practically the same R^2, which is the price of forcing the
zero crossing.

Usage::

    uv run sysid/direct/static_load.py                # all four joints
    uv run sysid/direct/static_load.py --joint joint_01
    SPIROB_FIG_LOCALE=de uv run sysid/direct/static_load.py   # thesis figures

Inputs : data/static_load/joint_*.csv, data/static_load/lever_arms.csv
Outputs: build/static_load/static_stiffness_<joint>.{pdf,png}
         build/static_load/static_stiffness_summary.csv
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from spirob import plotstyle as ts
from spirob.paths import STATIC_LOAD_DIR, build_dir

G = 9.81  # m/s^2


def read_lever_arms() -> dict[str, float]:
    """Lever arm per joint, in metres."""
    out: dict[str, float] = {}
    with open(STATIC_LOAD_DIR / "lever_arms.csv") as fh:
        for row in csv.DictReader(fh):
            out[row["joint"]] = float(row["lever_arm_m"])
    return out


def read_series(joint: str) -> tuple[np.ndarray, np.ndarray]:
    """(mass in grams, deflection in degrees) for one joint, comments skipped."""
    mass, angle = [], []
    with open(STATIC_LOAD_DIR / f"{joint}.csv") as fh:
        rows = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        mass.append(float(row["mass_g"]))
        angle.append(float(row["angle_deg"]))
    return np.array(mass), np.array(angle)


def evaluate(mass_g: np.ndarray, angle_deg: np.ndarray, r: float) -> dict:
    """Both regressions for one measurement series."""
    torque = mass_g / 1000.0 * G * r          # N·m
    phi = np.deg2rad(angle_deg)               # rad

    fit = linregress(phi, torque)
    residuals = torque - (fit.intercept + fit.slope * phi)

    k0 = float(np.sum(phi * torque) / np.sum(phi**2))   # through origin
    res0 = torque - k0 * phi
    ss_tot = float(np.sum((torque - torque.mean()) ** 2))

    return {
        "phi": phi,
        "torque": torque,
        "k": float(fit.slope),
        "M0": float(fit.intercept),
        "k_stderr": float(fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "k_origin": k0,
        "r_squared_origin": 1.0 - float(np.sum(res0**2)) / ss_tot,
        "n": len(mass_g),
        "r": r,
    }


def plot(joint: str, res: dict, out_stem) -> None:
    ts.apply_style()
    fig, ax = plt.subplots(figsize=(ts.FIG_FULL, 3.6))

    fit_x = np.linspace(0.0, res["phi"].max() * 1.04, 200)
    ax.plot(fit_x, res["M0"] + res["k"] * fit_x, label="Linear regression", **ts.model_kw())
    ax.plot(res["phi"], res["torque"], label="Measurements", zorder=5, **ts.marker_kw(0, size=6.5))

    ts.grid_on(ax)
    ax.set_xlabel(r"Deflection angle $\varphi$ (rad)")
    ax.set_ylabel(r"Torque $M$ (N·m)")
    ax.set_xlim(-0.01, fit_x.max())

    ts.annotate(
        ax,
        "\n".join(
            [
                joint.replace("_", " "),
                f"$k$ = {ts.num(res['k'], 4)} N·m/rad",
                f"$R^2$ = {ts.num(res['r_squared'], 4)}",
                f"$r$ = {ts.num(res['r'] * 100, 1)} cm,  $n$ = {res['n']}",
            ]
        ),
        loc="upper left",
    )
    ax.legend(loc="lower right")

    ax_deg = ax.secondary_xaxis("top", functions=(np.rad2deg, np.deg2rad))
    ax_deg.set_xlabel(r"Deflection angle $\varphi$ (°)", labelpad=6)
    ax_deg.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)

    ts.localize_axes(fig)
    ts.save(fig, str(out_stem))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--joint", action="append", help="Joint to evaluate (repeatable). Default: all.")
    args = ap.parse_args()

    levers = read_lever_arms()
    joints = args.joint or sorted(levers)
    out_dir = build_dir("static_load")

    rows = []
    for joint in joints:
        if joint not in levers:
            print(f"  unknown joint {joint!r} — known: {sorted(levers)}")
            continue
        mass_g, angle_deg = read_series(joint)
        res = evaluate(mass_g, angle_deg, levers[joint])

        print(f"\n=== {joint} (r = {res['r']:.3f} m, n = {res['n']}) ===")
        print(f"  torsional stiffness k = {res['k']:.5f} ± {res['k_stderr']:.5f} N·m/rad")
        print(f"  intercept M0          = {res['M0']:+.5f} N·m   (not shown in the figure)")
        print(f"  R^2                   = {res['r_squared']:.5f}")
        print(f"  RMSE of residuals     = {res['rmse']:.6f} N·m")
        print(f"  through-origin fit    : k = {res['k_origin']:.5f} N·m/rad, "
              f"R^2 = {res['r_squared_origin']:.5f}")

        plot(joint, res, out_dir / f"static_stiffness_{joint}")
        rows.append(
            {
                "joint": joint,
                "lever_arm_m": res["r"],
                "n_points": res["n"],
                "k_Nm_rad": round(res["k"], 5),
                "k_stderr": round(res["k_stderr"], 5),
                "M0_Nm": round(res["M0"], 5),
                "r_squared": round(res["r_squared"], 5),
                "rmse_Nm": round(res["rmse"], 6),
                "k_origin_Nm_rad": round(res["k_origin"], 5),
            }
        )

    if rows:
        summary = out_dir / "static_stiffness_summary.csv"
        with open(summary, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
