"""Publication-quality figures for the sim-to-sim identification study.

Reads the result JSONs written by ``sim2sim.py`` and produces three figures:

    fig_de_vs_cma.{pdf,png}         Differential Evolution vs CMA-ES
    fig_sensitivity.{pdf,png}       which parameter moves the trajectory
    fig_identifiability.{pdf,png}   how much information the trajectory carries

Usage::

    uv run sysid/simulation_based/sim2sim.py --compare --workers 10 --tol 1e-8
    uv run sysid/simulation_based/sim2sim.py --sensitivity --workers 10
    uv run sysid/figures/fig_sim2sim.py

Without a fresh run the script falls back to the shipped results in
``data/identified/``, so the figures reproduce from a clean clone.

PDF is the target format (vector, scales losslessly); the PNG is for quick
viewing. Set ``SPIROB_FIG_LOCALE=de`` for German decimal commas.

Outputs: build/figures/fig_*.{pdf,png}
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from spirob.paths import IDENTIFIED_DIR
from spirob.paths import build_dir as _build_dir
from spirob.plotstyle import _DECIMAL as DECIMAL

BUILD = _build_dir("figures")
RESULTS = _build_dir("sim2sim")

# ── Colour and chrome tokens (validated categorical palette, fixed slots) ──
C_SLOT1 = "#2a78d6"   # blau
C_SLOT2 = "#eb6834"   # orange
C_SLOT3 = "#1baf7a"   # aqua

C_DE = C_SLOT1
C_CMA = C_SLOT2
C_GROUP = {"stiffness": C_SLOT1, "damping": C_SLOT2, "tendon_stiffness": C_SLOT3}

INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis labels
GRID = "#e1e0d9"       # grid line
BASELINE = "#c3c2b7"   # axis line

LABEL = {
    "stiffness": "Joint stiffness",
    "damping": "Joint damping",
    "tendon_stiffness": "Tendon stiffness",
}


def apply_style() -> None:
    """Quiet chrome: thin spines, fine grid, text in ink rather than series colour."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.0,
        "figure.dpi": 120,
    })


def de_num(v: float, digits: int = 2) -> str:
    """Fixed-point number in the active decimal separator."""
    return f"{v:.{digits}f}".replace(".", DECIMAL)


def de_sci(v: float) -> str:
    """Power of ten as a superscript, e.g. 1e-06 -> 10⁻⁶."""
    exp = int(round(np.log10(v)))
    sup = str(exp).replace("-", "⁻")
    for a, b in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"):
        sup = sup.replace(a, b)
    return f"10{sup}"


def german_axes(fig, skip=(), skip_x=()) -> None:
    """Apply the decimal separator to every axis of the figure.

    ``skip`` leaves whole axes alone, ``skip_x`` only their x-axis -- needed
    for categorical axes whose ticks are already text.
    """
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos):
        if abs(v) >= 1000 and float(v).is_integer():
            return f"{int(v):,}".replace(",", " ")   # thin space
        return f"{v:g}".replace(".", DECIMAL)

    for ax in fig.axes:
        if ax in skip:
            continue
        axes_pairs = [(ax.yaxis, False)]
        if ax not in skip_x:
            axes_pairs.append((ax.xaxis, False))
        for axis, _ in axes_pairs:
            if axis.get_scale() == "linear":
                axis.set_major_formatter(FuncFormatter(fmt))


def grid_on(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, linestyle="-", alpha=1.0, zorder=0)
    ax.set_axisbelow(True)


def scenario_label(sim_time: float, angle: float) -> str:
    pose = "rest pose" if angle == 0 else f"{angle:.0f}° deflected"
    return f"{sim_time:.0f} s, {pose}"


def save(fig, stem: str) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = BUILD / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  [Info] {p}")
    plt.close(fig)


# ====================================================================
#  Figure 1 -- DE vs CMA-ES
# ====================================================================

def figure_de_vs_cma(data: dict) -> None:
    records = data["records"]
    # For the per-scenario detail panels, take the run with more iterations
    scenarios = []
    for r in records:
        key = (r["sim_time"], r["init_angle_deg"])
        if key not in scenarios:
            scenarios.append(key)
    scenarios.sort(key=lambda k: (k[1], k[0]))

    def pick(sim_time, angle, opt, maxiter=400):
        for r in records:
            if (r["sim_time"], r["init_angle_deg"], r["optimizer"], r["maxiter"]) == \
               (sim_time, angle, opt, maxiter):
                return r
        return None

    fig, axs = plt.subplots(2, 3, figsize=(11.0, 6.6))

    # ── Row 1: convergence per scenario ──────────────────────────────
    for col, (st, ang) in enumerate(scenarios):
        ax = axs[0][col]
        for opt, color in (("de", C_DE), ("cma", C_CMA)):
            r = pick(st, ang, opt)
            if r is None:
                continue
            hist = np.array(r["cost_history"], dtype=float)
            # Cumulative evaluations: the first point sits after the FIRST
            # generation, not after the first evaluation (a DE generation = 560)
            x = r["nfev"] * np.arange(1, len(hist) + 1) / len(hist)
            ax.plot(x, hist, color=color, linewidth=2.0,
                    label="Differential Evolution" if opt == "de" else "CMA-ES",
                    solid_capstyle="round")
            # Endpunkt direkt markieren statt jeden Punkt zu beschriften
            ax.plot(x[-1], hist[-1], marker="o", markersize=5, color=color,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"({'abc'[col]}) {scenario_label(st, ang)}")
        ax.set_xlabel("Function evaluations")
        if col == 0:
            ax.set_ylabel("Best cost")
        grid_on(ax)

    # ── Row 2: final cost, effort, parameter error ───────────────────
    labels = [scenario_label(st, ang) for st, ang in scenarios]
    xpos = np.arange(len(scenarios), dtype=float)
    width = 0.36

    def grouped_bars(ax, values_de, values_cma, ylabel, logy=False, fmt=lambda v: de_num(v, 2)):
        b1 = ax.bar(xpos - width / 2, values_de, width * 0.92, color=C_DE,
                    label="Differential Evolution", zorder=3)
        b2 = ax.bar(xpos + width / 2, values_cma, width * 0.92, color=C_CMA,
                    label="CMA-ES", zorder=3)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels([lab.replace(", ", "\n") for lab in labels])
        ax.set_ylabel(ylabel)
        grid_on(ax, axis="y")
        for bars in (b1, b2):
            for bar in bars:
                v = bar.get_height()
                ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7, color=INK_2)
        return b1, b2

    ax = axs[1][0]
    de_vals = [pick(st, a, "de")["cost"] for st, a in scenarios]
    cma_vals = [pick(st, a, "cma")["cost"] for st, a in scenarios]
    grouped_bars(ax, de_vals, cma_vals, "Final cost", fmt=lambda v: de_num(v, 3))
    ax.set_title("(d) Final quality")

    ax = axs[1][1]
    de_vals = [pick(st, a, "de")["nfev"] for st, a in scenarios]
    cma_vals = [pick(st, a, "cma")["nfev"] for st, a in scenarios]
    grouped_bars(ax, de_vals, cma_vals, "Function evaluations",
                 fmt=lambda v: f"{v:,.0f}".replace(",", "\u2009"))
    ax.set_title("(e) Effort until termination")

    ax = axs[1][2]
    groups = ["stiffness", "damping", "tendon_stiffness"]
    gx = np.arange(len(groups), dtype=float)
    de_m, cma_m = [], []
    for g in groups:
        de_m.append(np.mean([pick(st, a, "de")[f"{g.replace('tendon_stiffness','tendon')}_mape"]
                             for st, a in scenarios]))
        cma_m.append(np.mean([pick(st, a, "cma")[f"{g.replace('tendon_stiffness','tendon')}_mape"]
                              for st, a in scenarios]))
    b1 = ax.bar(gx - width / 2, de_m, width * 0.92, color=C_DE, zorder=3)
    b2 = ax.bar(gx + width / 2, cma_m, width * 0.92, color=C_CMA, zorder=3)
    ax.set_xticks(gx)
    ax.set_xticklabels([LABEL[g].replace(" ", "\n")
                        for g in groups])
    ax.set_ylabel("Parameter error (MAPE) [%]")
    ax.set_title("(f) Accuracy, averaged over the scenarios")
    grid_on(ax, axis="y")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=INK_2)

    handles = [Line2D([], [], color=C_DE, linewidth=3, label="Differential Evolution"),
               Line2D([], [], color=C_CMA, linewidth=3, label="CMA-ES")]
    german_axes(fig, skip_x=[axs[1][0], axs[1][1], axs[1][2]])
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_de_vs_cma")




# ====================================================================
#  Figure 2 -- parameter sensitivity
# ====================================================================

def _pick_scenario(scen: list[dict], sim_time: float, angle: float) -> dict:
    for s in scen:
        if s["sim_time"] == sim_time and s["init_angle_deg"] == angle:
            return s
    return scen[0]


def figure_sensitivity(sens: dict, comp: dict | None) -> None:
    """Which parameter moves the observable, and by how much?"""
    scen = sens["scenarios"]
    # The 4 s scenario is the only one in which the cost function still
    # carries parameter information (see figure 3) -- hence the reference.
    main = _pick_scenario(scen, 4.0, 0.0)

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9))

    # ── (a) Cost rise over relative deviation, log scaled ────────────
    ax = axs[0]
    dev = np.array(main["dev_log"], dtype=float) * 100.0
    for g in ("stiffness", "damping", "tendon_stiffness"):
        ax.plot(dev, main["oat_log"][g], color=C_GROUP[g], linewidth=2.0, label=LABEL[g])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Deviation from the true value [%]")
    ax.set_ylabel("Cost")
    ax.set_title("(a) Cost increase per parameter group")
    grid_on(ax)
    ax.legend(loc="upper left")

    # ── (b) Group sensitivity per scenario ───────────────────────────
    ax = axs[1]
    groups = ("stiffness", "damping", "tendon_stiffness")
    xs = np.arange(len(scen), dtype=float)
    w = 0.26
    for k, g in enumerate(groups):
        vals = [s["group_sens"][g] for s in scen]
        bars = ax.bar(xs + (k - 1) * w, vals, w * 0.9, color=C_GROUP[g], label=LABEL[g], zorder=3)
        for bar in bars:
            ax.annotate(de_num(bar.get_height(), 2),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=7, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([scenario_label(s["sim_time"], s["init_angle_deg"]).replace(", ", "\n")
                        for s in scen])
    ax.set_ylabel("Trajectory change [°]")
    ax.set_title(f"(b) Effect of a {sens['scenarios'][0]['delta']*100:.0f} % parameter change")
    grid_on(ax, axis="y")
    ax.legend(loc="upper left", ncol=1)

    # ── (c) Sensitivity per joint ────────────────────────────────────
    ax = axs[2]
    for g in ("stiffness", "damping"):
        v = np.array(main["traj_sens"][g], dtype=float)
        ax.plot(np.arange(len(v)), v, color=C_GROUP[g], linewidth=2.0, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=0.8, label=LABEL[g])
    v_t = float(np.mean(main["traj_sens"]["tendon_stiffness"]))
    ax.axhline(v_t, color=C_GROUP["tendon_stiffness"], linewidth=2.0, linestyle=(0, (5, 2)),
               label=LABEL["tendon_stiffness"] + " (mean)")
    ax.set_xlabel("Joint index (0 = base, 12 = tip)")
    ax.set_ylabel("Trajectory change [°]")
    ax.set_ylim(bottom=0)
    ax.set_title(f"(c) Distribution across the joints, {scenario_label(main['sim_time'], main['init_angle_deg'])}")
    grid_on(ax)
    ax.legend(loc="lower right")

    german_axes(fig, skip_x=[axs[1]])
    fig.tight_layout()
    save(fig, "fig_sensitivity")


# ====================================================================
#  Figure 3 -- identifiability / limits of the cost function
# ====================================================================

def figure_identifiability(sens: dict, comp: dict | None) -> None:
    scen = sens["scenarios"]
    long_s = _pick_scenario(scen, 20.0, 0.0)
    short_s = _pick_scenario(scen, 4.0, 0.0)

    fig, axs = plt.subplots(1, 3, figsize=(12.4, 4.0))

    # ── (a) Trajectory divergence under tiny perturbations ───────────
    ax = axs[0]
    t = np.array(long_s["divergence_time"], dtype=float)
    order = sorted(long_s["divergence"].keys(), key=float)
    shades = [C_SLOT1, C_SLOT2, C_SLOT3]
    for c, key in zip(shades, order):
        y = np.array(long_s["divergence"][key], dtype=float)
        y = np.maximum(y, 1e-9)
        ax.plot(t, y, color=c, linewidth=2.0,
                label=f"perturbation {de_sci(float(key))}")
    # Saturation level = mean of the last quarter of the strongest perturbation
    sat = float(np.mean(np.array(long_s["divergence"][order[-1]], dtype=float)[-len(t) // 4:]))
    ax.axhline(sat, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"decorrelation level ≈ {de_num(sat, 2)}°", (t[1], sat),
                textcoords="offset points", xytext=(2, 5), fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle deviation (RMS) [°]")
    ax.set_title("(a) Divergence under a minimal parameter perturbation")
    grid_on(ax)
    ax.legend(loc="lower right", title="relative parameter change",
              title_fontsize=7.5)

    # ── (b) Cost over the length of the fitting horizon ──────────────
    ax = axs[1]
    th = np.array(long_s["horizon_time"], dtype=float)
    for c, key in zip(shades, sorted(long_s["horizon_costs"].keys(), key=float)):
        y = np.array(long_s["horizon_costs"][key], dtype=float)
        ax.plot(th, y, color=c, linewidth=2.0, label=f"perturbation {de_sci(float(key))}")
    # Usable horizon: up to here the cost function still separates a small from
    # still differ by at least a factor of 2 for a 100x larger deviation.
    keys = sorted(long_s["horizon_costs"].keys(), key=float)
    c_small = np.array(long_s["horizon_costs"][keys[0]], dtype=float)
    c_large = np.array(long_s["horizon_costs"][keys[-1]], dtype=float)
    sep = c_small < 0.5 * c_large
    if sep.any():
        t_use = float(th[np.max(np.flatnonzero(sep))])
        ax.axvline(t_use, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate(f"usable horizon ≈ {de_num(t_use, 1)} s", (t_use, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -10), fontsize=7.5, color=INK_2)
    ax.set_xlabel("Length of the evaluated time window [s]")
    ax.set_ylabel("Cost")
    ax.set_title("(b) Information loss with window length")
    grid_on(ax)
    ax.legend(loc="lower right", title="relative parameter change",
              title_fontsize=7.5)

    # ── (c) Cost landscape with the solutions that were found ────────
    ax = axs[2]
    gf = np.array(short_s["landscape_factors"], dtype=float)
    L = np.array(short_s["landscape"], dtype=float)
    floor = max(float(L[L > 0].min()), 1e-3)
    Lp = np.maximum(L, floor)
    levels = np.logspace(np.log10(floor), np.log10(Lp.max()), 22)
    cf = ax.contourf(gf, gf, Lp.T, levels=levels, norm=LogNorm(), cmap="Blues", zorder=2)
    cs = ax.contour(gf, gf, Lp.T, levels=[0.05, 0.1, 0.2], colors="white",
                    linewidths=0.8, zorder=3)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt=lambda v: de_num(v, 2))
    ax.plot(1.0, 1.0, marker="*", markersize=14, color="white", markeredgecolor=INK,
            markeredgewidth=0.9, zorder=6)
    ax.annotate("true value", (1.0, 1.0), textcoords="offset points", xytext=(9, 5),
                fontsize=7.5, color=INK)

    if comp is not None:
        gt_s = float(np.mean(short_s["ground_truth"]["stiffness"]))
        gt_d = float(np.mean(short_s["ground_truth"]["damping"]))
        for opt, color, mk, lbl in (("de", C_DE, "o", "DE solution"),
                                    ("cma", C_CMA, "s", "CMA-ES solution")):
            done = False
            for r in comp["records"]:
                if (r["optimizer"], r["sim_time"], r["init_angle_deg"]) != \
                   (opt, short_s["sim_time"], short_s["init_angle_deg"]):
                    continue
                fs = float(np.mean(r["identified"]["stiffness"])) / gt_s
                fd = float(np.mean(r["identified"]["damping"])) / gt_d
                ax.plot(fs, fd, marker=mk, markersize=7, color=color, linestyle="none",
                        markeredgecolor="white", markeredgewidth=1.2, zorder=7,
                        label=None if done else lbl)
                done = True
        ax.legend(loc="upper left", labelcolor=INK_2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    for setter, ticker in ((ax.set_xticks, ax.set_xticklabels), (ax.set_yticks, ax.set_yticklabels)):
        setter([0.5, 1.0, 2.0])
        ticker([de_num(0.5, 1), "1", "2"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(gf.min(), gf.max())
    ax.set_ylim(gf.min(), gf.max())
    ax.set_xlabel("Factor on the joint stiffness")
    ax.set_ylabel("Factor on the joint damping")
    ax.set_title(f"(c) Cost landscape, {scenario_label(short_s['sim_time'], short_s['init_angle_deg'])}")
    cb = fig.colorbar(cf, ax=ax, pad=0.02, ticks=[0.01, 0.03, 0.1, 0.3])
    cb.ax.set_yticklabels([de_num(v, 2) for v in (0.01, 0.03, 0.1, 0.3)])
    cb.set_label("Cost", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, color=MUTED)
    cb.outline.set_edgecolor(BASELINE)

    german_axes(fig, skip=[axs[2]])
    fig.tight_layout()
    save(fig, "fig_identifiability")


def main() -> None:
    apply_style()
    comp_path = RESULTS / "sysid_multi_de_vs_cma.json"
    if not comp_path.exists():
        comp_path = IDENTIFIED_DIR / "sim2sim_de_vs_cma.json"
    sens_path = RESULTS / "sysid_sensitivity.json"
    if not sens_path.exists():
        sens_path = IDENTIFIED_DIR / "sim2sim_sensitivity.json"

    comp = json.loads(comp_path.read_text()) if comp_path.exists() else None
    if comp is not None:
        print("Figure 1 — DE vs CMA-ES")
        figure_de_vs_cma(comp)
    else:
        print(f"  [Warn] {comp_path} missing — figure 1 skipped")

    if sens_path.exists():
        sens = json.loads(sens_path.read_text())
        print("Figure 2 — sensitivity")
        figure_sensitivity(sens, comp)
        print("Figure 3 — identifiability")
        figure_identifiability(sens, comp)
    else:
        print(f"  [Warn] {sens_path} missing — figures 2/3 skipped")


if __name__ == "__main__":
    main()
