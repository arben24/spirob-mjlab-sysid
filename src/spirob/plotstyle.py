"""Shared figure style for every SpiRob figure.

One style across all sys-id scripts so the figures read as a series. Three
requirements drive the design:

1. **Colour-vision deficiency** — the categorical colour slots come from a
   palette checked against deuteranopia/protanopia/tritanopia; the first three
   slots keep their separation in any pairing.
2. **Greyscale print** — identity never rests on colour alone. Each slot also
   carries its own dash pattern, marker and hatch (:func:`line_kw`,
   :func:`marker_kw`, :func:`bar_kw`) — secondary encoding only where several
   series share *one* panel, never as decoration.
3. **Quiet chrome** — thin spines, fine grid, text in ink rather than in the
   series colour; the data is the loudest thing in the frame.

Usage::

    from spirob import plotstyle as ts

    ts.apply_style()
    fig, ax = plt.subplots(figsize=(ts.FIG_FULL, 3.4))
    ax.plot(x, y, label="measurement", **ts.line_kw(0))
    ts.grid_on(ax)
    ts.localize_axes(fig)
    ts.save(fig, "build/my_figure")   # writes .pdf and .png

Set ``SPIROB_FIG_LOCALE=de`` for the thesis figures (decimal comma).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── Figure widths (inches) ──────────────────────────────────────────────
# Sized for a ~16 cm text block: the figure is embedded with
# \includegraphics[width=\textwidth]{...} without scaling, so the font sizes
# below match the document's body text.
FIG_FULL = 6.30   # full text width (16 cm)
FIG_WIDE = 7.20   # wide figure (landscape / appendix)
FIG_HALF = 3.05   # half text width, two figures side by side

# ── Categorical colour slots (fixed order, never extended cyclically) ───
# The order is verified, not guessed: for all ten pairings of the five slots
# the OKLab distance (x100) exceeds the target of 8 -- also after
# simulating the three dichromacies (Vienot projection in LMS space):
#   normal vision  worst pair dE 13.7  (orange/yellow)
#   protanopia     worst pair dE 10.0  (orange/aqua)
#   deuteranopia   worst pair dE  8.8  (orange/yellow)
#   tritanopia     worst pair dE 10.5  (blue/aqua)
# A sixth slot (magenta) would fall below that (dE 6.0 against yellow under
# tritanopia) and is deliberately absent: more than five series should be
# grouped, or split across several panels.
C1 = "#2a78d6"   # blue      (luminance 0.19)
C2 = "#eb6834"   # orange    (0.28)
C3 = "#1baf7a"   # aqua      (0.32)
C4 = "#4a3aa7"   # violet    (0.07)
C5 = "#eda100"   # gelb      (0,44)
PALETTE = (C1, C2, C3, C4, C5)

# ── Chrome-Token ────────────────────────────────────────────────────────
INK = "#0b0b0b"        # primary text, titles
INK_2 = "#52514e"      # secondary text, axis labels, model curves
MUTED = "#898781"      # Ticks
GRID = "#e1e0d9"       # Gitterlinie
BASELINE = "#c3c2b7"   # Achsenlinie
FAINT = "#d7d6cf"      # Rohsignal, Hintergrundspuren
SURFACE = "white"      # drawing surface (separating edges between marks)

# ── Status and control-element colours ──────────────────────────────────
# Reserved: never use as a series colour. Status is always shown together with
# text or a symbol, never by colour alone.
GOOD = "#1e7a4c"       # success, saved
GOOD_BG = "#e6f2ec"
WARN = "#b3261e"       # warning, no valid evaluation
WARN_BG = "#fbeae8"

# Surfaces for the GUI figures: control panels lift off the page without
# competing with the data for attention.
UI_PANEL = "#f4f3ee"        # background of a control-panel group
UI_PANEL_EDGE = "#dedcd2"
UI_FACE = "#e6e5dd"         # button, idle
UI_FACE_HOVER = "#d3d2c8"
UI_ACCENT = C1              # primary action (save)
UI_ACCENT_HOVER = "#5596e2"
UI_DANGER = "#e9a79f"       # destructive action (delete)
UI_DANGER_HOVER = "#dd8478"

# ── Secondary encoding per slot (carries identity in greyscale print) ───
DASHES: tuple[Any, ...] = (
    (0, ()),                              # durchgezogen
    (0, (5.0, 1.8)),                      # gestrichelt
    (0, (4.0, 1.4, 1.0, 1.4)),            # Strichpunkt
    (0, (1.0, 1.6)),                      # gepunktet
    (0, (7.0, 1.6, 1.0, 1.6, 1.0, 1.6)),  # lang-Punkt-Punkt
)
MARKERS = ("o", "s", "^", "D", "v")
HATCHES = ("", "///", "\\\\\\", "...", "xxx")


def _slot(i: int) -> int:
    """Slot index; beyond the palette we deliberately do not cycle colours."""
    if i >= len(PALETTE):
        raise ValueError(
            f"Slot {i} does not exist -- the palette has {len(PALETTE)} slots. "
            "Group the extra series, or split them across several panels."
        )
    return i


def color(i: int) -> str:
    """Colour of the i-th slot."""
    return PALETTE[_slot(i)]


def line_kw(i: int, width: float = 1.6, marker: bool = False,
            **extra: Any) -> dict[str, Any]:
    """Line style for series ``i`` -- colour **and** dash pattern."""
    kw: dict[str, Any] = {
        "color": PALETTE[_slot(i)],
        "linestyle": DASHES[i],
        "linewidth": width,
        "solid_capstyle": "round",
        "dash_capstyle": "round",
    }
    if marker:
        kw.update({
            "marker": MARKERS[i],
            "markersize": 4.5,
            "markeredgecolor": SURFACE,
            "markeredgewidth": 0.8,
            "markevery": 0.12,
        })
    kw.update(extra)
    return kw


def marker_kw(i: int, size: float = 6.0, **extra: Any) -> dict[str, Any]:
    """Marker style for scatter plots -- colour **and** marker shape.

    The light edge around each mark keeps overlapping points separable in print.
    """
    kw: dict[str, Any] = {
        "color": PALETTE[_slot(i)],
        "marker": MARKERS[i],
        "markersize": size,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 0.9,
        "linestyle": "none",
    }
    kw.update(extra)
    return kw


def bar_kw(i: int, **extra: Any) -> dict[str, Any]:
    """Bar style -- colour **and** hatch, with an edge for separation."""
    kw: dict[str, Any] = {
        "color": PALETTE[_slot(i)],
        "hatch": HATCHES[i],
        "edgecolor": SURFACE,
        "linewidth": 0.8,
        "zorder": 3,
    }
    kw.update(extra)
    return kw


def model_kw(width: float = 1.4, **extra: Any) -> dict[str, Any]:
    """Style for model/fit curves: dark grey, dashed.

    Deliberately neutral, so it keeps its contrast against every series colour,
    including in greyscale print.
    """
    kw: dict[str, Any] = {
        "color": INK_2,
        "linestyle": (0, (4.5, 1.8)),
        "linewidth": width,
        "dash_capstyle": "round",
        "zorder": 4,
    }
    kw.update(extra)
    return kw


def apply_style(scale: float = 1.0) -> None:
    """Set the rcParams. Call once per process, before the first plot.

    ``scale`` enlarges every font size together. For printed figures 1.0 is
    right (the figure is embedded at its original size); for screen captures
    that are later scaled down to text width, a value > 1 is needed so labels
    stay readable in the document.
    """
    def sz(v: float) -> float:
        return round(v * scale, 2)

    plt.rcParams.update({
        # Surfaces
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Type -- sized for \textwidth embedding without scaling
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": sz(9),
        "text.color": INK,
        "axes.titlesize": sz(9.5),
        "axes.titlecolor": INK,
        "axes.titleweight": "regular",
        "axes.labelsize": sz(9),
        "axes.labelcolor": INK_2,
        "xtick.labelsize": sz(8),
        "ytick.labelsize": sz(8),
        "legend.fontsize": sz(8),
        "figure.titlesize": sz(10.5),
        # Achsen
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # Gitter
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        # Marken
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.8,
        "hatch.linewidth": 0.6,
        "hatch.color": SURFACE,
        # Legende
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.4,
        # Output
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,     # embed TrueType rather than Type-3
        "ps.fonttype": 42,
    })


# ── Number formatting ───────────────────────────────────────────────────
#
# The repository publishes in English (decimal point); the thesis figures are
# German (decimal comma). One switch decides, so the same script produces both:
#
#     SPIROB_FIG_LOCALE=de  uv run sysid/direct/static_load.py
#
LOCALE = os.environ.get("SPIROB_FIG_LOCALE", "en").lower()

_DECIMAL = "," if LOCALE.startswith("de") else "."


def num(v: float, digits: int = 2) -> str:
    """Fixed-point number in the active locale's decimal separator."""
    return f"{v:.{digits}f}".replace(".", _DECIMAL)


def de_num(v: float, digits: int = 2) -> str:
    """Backwards-compatible alias of :func:`num`."""
    return num(v, digits)


def sci(v: float, digits: int = 2) -> str:
    """Scientific notation with a superscript exponent, e.g. 1.23·10⁻³."""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    mant = v / 10.0 ** exp
    sup = str(exp).replace("-", "⁻")
    for a, b in zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"):
        sup = sup.replace(a, b)
    return f"{num(mant, digits)}·10{sup}"


def de_sci(v: float, digits: int = 2) -> str:
    """Backwards-compatible alias of :func:`sci`."""
    return sci(v, digits)


def auto(v: float, sig: int = 3) -> str:
    """Fixed-point inside a readable range, scientific notation outside it."""
    if v == 0:
        return "0"
    a = abs(v)
    if 0.01 <= a < 10000:
        digits = max(0, sig - 1 - int(np.floor(np.log10(a))))
        return num(v, digits)
    return sci(v, 2)


def de_auto(v: float, sig: int = 3) -> str:
    """Backwards-compatible alias of :func:`auto`."""
    return auto(v, sig)


def localize_axes(fig, skip: Iterable = (), skip_x: Iterable = ()) -> None:
    """Apply the locale's decimal separator to every linear axis of the figure.

    ``skip`` leaves whole axes alone, ``skip_x`` only their x-axis — needed for
    categorical axes whose ticks are already text.
    """
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos):
        if abs(v) >= 1000 and float(v).is_integer():
            return f"{int(v):,}".replace(",", " ")   # schmales Leerzeichen
        return f"{v:g}".replace(".", _DECIMAL)

    skip = list(skip)
    skip_x = list(skip_x)
    for ax in fig.axes:
        if ax in skip:
            continue
        axes = [ax.yaxis] + ([] if ax in skip_x else [ax.xaxis])
        for axis in axes:
            if axis.get_scale() == "linear":
                axis.set_major_formatter(FuncFormatter(fmt))


def german_axes(fig, skip: Iterable = (), skip_x: Iterable = ()) -> None:
    """Backwards-compatible alias of :func:`localize_axes`."""
    localize_axes(fig, skip=skip, skip_x=skip_x)


# ── Hilfen ──────────────────────────────────────────────────────────────

def grid_on(ax, axis: str = "both") -> None:
    """A fine grid behind the data."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)


def annotate(ax, text: str, loc: str = "upper left", **extra: Any):
    """Metrics as an unobtrusive text block inside the axes."""
    corners = {
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }
    x, y, ha, va = corners[loc]
    kw: dict[str, Any] = {
        "transform": ax.transAxes, "ha": ha, "va": va,
        "fontsize": 8, "color": INK_2, "linespacing": 1.5,
        "bbox": dict(facecolor=SURFACE, edgecolor=BASELINE,
                     linewidth=0.6, boxstyle="round,pad=0.4", alpha=0.92),
        "zorder": 6,
    }
    kw.update(extra)
    return ax.text(x, y, text, **kw)


def value_labels(ax, xs: Sequence[float], ys: Sequence[float],
                 texts: Sequence[str], offset: float = 0.02) -> None:
    """Direct value labels above bars -- text in ink, not in the series colour."""
    lo, hi = ax.get_ylim()
    pad = (hi - lo) * offset
    for x, y, t in zip(xs, ys, texts):
        ax.text(x, y + pad, t, ha="center", va="bottom",
                fontsize=7.5, color=INK_2, zorder=5)


def save(fig, stem: str, formats: Sequence[str] = ("pdf", "png"),
         quiet: bool = False) -> list[str]:
    """Save the figure as a vector PDF (for the document) and a PNG (for viewing).

    ``stem`` may be given with or without an extension.
    """
    root, ext = os.path.splitext(stem)
    if ext.lower().lstrip(".") not in ("pdf", "png", "svg", ""):
        root = stem
    directory = os.path.dirname(root)
    if directory:
        os.makedirs(directory, exist_ok=True)
    paths = []
    for f in formats:
        p = f"{root}.{f}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
        if not quiet:
            print(f"  → {p}")
    plt.close(fig)
    return paths
