"""Publication figures of a workspace sweep: where in the shell the policy is accurate.

Reads one ``.npz`` written by :mod:`spirob_rl.rig.workspace_sweep` and draws it in the
same visual language as ``spirob/target_figure.py`` and
``inverted_pendulum/sysid_figure.py`` -- vector output, white background, thick
strokes, print-sized type, and a second non-colour identity (marker, dash
pattern) for every line so the figure survives greyscale.

Two figures, both from the same file:

``karte``
    The map itself: the trained shell, filled cell by cell with what the sweep
    measured. Drawn as a ``pcolormesh`` in the shell's own polar coordinates, so
    the cells follow the arc instead of being squeezed into a rectangle that is
    mostly empty. Four panels by default (``--panels``):

    * ``error`` -- residual TCP-target distance after settling.
    * ``success`` -- how many of the start postures got below the threshold.
      ``--threshold`` re-asks this question at another distance without
      re-running anything; the raw distances are in the file.
    * ``spread`` -- the standard deviation across start postures, and the panel
      the paired design exists for: where it is large, the *start posture*
      decides the outcome, not the target.
    * ``deviation`` -- not how far off but *which way*: an arrow from every
      target to the median resting place of the tip. A map of magnitudes cannot
      separate a policy that scatters from one that is systematically short.

``profil``
    The two one-dimensional cuts the map invites: error along the shell (one
    line per radial layer, so "the inner band is harder" becomes visible as
    such) and the approach over time as a median with a 10-90 % band.

Colour carries one meaning throughout: **yellow is good, dark is bad**. Error
and spread use ``viridis_r`` (low = yellow), success rate ``viridis`` (high =
yellow), so a reader who learns the scale once reads every panel. Both are
perceptually uniform, colour-vision-deficiency safe, and monotonic in greyscale.

Usage::

    uv run python -m spirob_rl.rig.workspace_figure --sweep build/rl/workspace/sweep_*.npz
    uv run python -m spirob_rl.rig.workspace_figure --sweep ... --threshold 30 --language en
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .target_figure import (
  _BASE_COLOR,
  _GRID_COLOR,
  _MUTED_COLOR,
  _default_language,
  apply_style,
)
from .workspace_sweep import ShellGeometry, SweepResult

# Okabe-Ito slots for the line plots, in the fixed order the other figure
# modules use them in -- the order is the colour-blindness safety mechanism and
# is not rearranged by taste.
_LAYER_COLORS = ("#0071b2", "#d55c00", "#009e74", "#8c3d8f", "#56b3e9")
_LAYER_MARKERS = ("o", "s", "^", "D", "v")
_LAYER_DASHES = ((0, ()), (0, (5, 2)), (0, (1.5, 1.5)), (0, (7, 2, 1.5, 2)))
_ARROW_COLOR = "#0071b2"
_THRESHOLD_COLOR = "#d55c00"

_COLOUR_PERCENTILE = 97.5
"""Where the error and spread colour scales end by default.

A handful of outlier cells would otherwise own most of the colormap and leave
the range everything else lives in as one indistinguishable green. The
colorbar is drawn with an arrow at that end, so nothing is hidden -- the cells
above it are simply "at least this bad". ``--vmax-error`` overrides it.
"""

_QUIVER_KEY_M = 0.05
"""Length of the reference arrow in the deviation panel [m]."""

_AXIS_LIMITS = (1.1, 1.12, 0.26)
"""x half-width, z top and z bottom of a workspace panel, in units of the reach."""

_AXIS_ASPECT = (_AXIS_LIMITS[1] + _AXIS_LIMITS[2]) / (2 * _AXIS_LIMITS[0])
"""Height/width of a workspace panel's data box; sets the figure height."""

_PANEL_DECORATION_W = 1.13
_PANEL_DECORATION_H = 0.60
_FIGURE_DECORATION_H = 0.30
"""Inches a workspace panel spends on chrome at scale 1.0 (axis labels, ticks,
title, colorbar). The panels are equal-aspect, so their drawn height follows
from their width -- and a figure sized without accounting for that does not get
a taller plot, it gets white space between the rows that ``tight_layout``
cannot take back out."""

LABELS = {
  "de": {
    "x": "x [m]",
    "z": "z [m]",
    "angle": "Winkel zur +z-Achse [rad]",
    "time": "Zeit seit Zielvorgabe [s]",
    "error": "Restabstand [mm]",
    "error_title": "Genauigkeit",
    "error_subtitle": "mittlerer Restabstand zum Ziel",
    "success_title": "Zuverlässigkeit",
    "success_subtitle": "Erfolg bei unter {threshold:.0f} mm",
    "success_cbar": "Anteil der Läufe [%]",
    "spread_title": "Einfluss der Startpose",
    "spread_subtitle": "Streuung zwischen den Läufen",
    "spread_cbar": "Standardabweichung [mm]",
    "deviation_title": "Richtung des Fehlers",
    "deviation_subtitle": "Ziel → Ruhelage der Spitze",
    "distance": "Abstand TCP–Ziel [mm]",
    "median": "Median",
    "band": "10–90 % der Läufe",
    "threshold": "Schwelle {threshold:.0f} mm",
    "layer": "Schale {deviation:+.0f} mm",
    "layer_centre": "Schalenmitte",
    "base": "Basis",
    "profile_angle": "Genauigkeit entlang der Schale",
    "profile_time": "Annäherung an das Ziel (alle Läufe)",
    "window": "Mittelungsfenster",
  },
  "en": {
    "x": "x [m]",
    "z": "z [m]",
    "angle": "angle from +z axis [rad]",
    "time": "time since target was set [s]",
    "error": "residual distance [mm]",
    "error_title": "accuracy",
    "error_subtitle": "mean residual distance to the target",
    "success_title": "reliability",
    "success_subtitle": "success below {threshold:.0f} mm",
    "success_cbar": "share of runs [%]",
    "spread_title": "influence of the start posture",
    "spread_subtitle": "spread between the runs",
    "spread_cbar": "standard deviation [mm]",
    "deviation_title": "direction of the error",
    "deviation_subtitle": "target → resting place of the tip",
    "distance": "TCP–target distance [mm]",
    "median": "median",
    "band": "10–90 % of runs",
    "threshold": "threshold {threshold:.0f} mm",
    "layer": "shell {deviation:+.0f} mm",
    "layer_centre": "shell centre",
    "base": "base",
    "profile_angle": "accuracy along the shell",
    "profile_time": "approach to the target (all runs)",
    "window": "averaging window",
  },
}

PANELS = ("error", "success", "spread", "deviation")
PROFILE_PANELS = ("angle", "time")
FIGURES = ("karte", "profil")


# -- geometry -----------------------------------------------------------------


def _edges(centres: np.ndarray, fallback_half_width: float) -> np.ndarray:
  """Cell edges around grid centres, clamped to the swept range.

  Clamping matters: without it the outermost half cell would stick out past the
  angle limit and the band, drawing a workspace slightly larger than the one
  that was actually measured.
  """
  if len(centres) == 1:
    return np.array([centres[0] - fallback_half_width, centres[0] + fallback_half_width])
  mid = 0.5 * (centres[:-1] + centres[1:])
  return np.concatenate([[centres[0]], mid, [centres[-1]]])


def cell_corners(result: SweepResult) -> tuple[np.ndarray, np.ndarray]:
  """(X, Z) corner grids for ``pcolormesh``, following the shell's arc.

  ``[n_angle + 1, n_radius + 1]`` each. A rectangular heatmap over (angle,
  deviation) would be easier to draw, but the question is *where in the
  workspace* -- so the cells are mapped back into the x-z plane the arm works
  in, and the figure has the shape of the thing it describes.
  """
  angle_edges = _edges(result.angles, result.geometry.angle_limit)
  dev_edges = _edges(result.deviations, result.geometry.shell_band)
  angle_grid, dev_grid = np.meshgrid(angle_edges, dev_edges, indexing="ij")
  xz = result.geometry.to_xz(angle_grid, dev_grid)
  return xz[..., 0], xz[..., 1]


def _draw_shell(axis, geometry: ShellGeometry, language: str, scale: float) -> None:
  """Chrome shared by every workspace panel: centreline, angle limits, base."""
  text = LABELS[language]
  angle = np.linspace(-geometry.angle_limit, geometry.angle_limit, 241)
  centre = geometry.to_xz(angle, np.zeros_like(angle))
  axis.plot(centre[:, 0], centre[:, 1], color=_MUTED_COLOR, linewidth=1.4,
            linestyle=(0, (6, 3)), zorder=4)
  for sign in (-1.0, 1.0):
    x, z = geometry.to_xz(np.array(sign * geometry.angle_limit), np.array(0.0))
    axis.plot([0.0, x * 1.2], [0.0, z * 1.2], color=_GRID_COLOR, linewidth=1.2,
              linestyle=(0, (2, 4)), zorder=1)
  axis.plot([0.0], [0.0], linestyle="none", marker="s", markersize=7 * scale,
            color=_BASE_COLOR, zorder=6)
  axis.annotate(text["base"], (0.0, 0.0), textcoords="offset points",
                xytext=(0.0, -8 * scale), ha="center", va="top",
                color=_BASE_COLOR, fontsize=9 * scale)


def _style_workspace_axis(
  axis, geometry: ShellGeometry, language: str, show_x: bool, show_y: bool
) -> None:
  text = LABELS[language]
  reach = geometry.shell_radius + geometry.shell_band
  axis.set_aspect("equal")
  axis.set_xlim(-reach * _AXIS_LIMITS[0], reach * _AXIS_LIMITS[0])
  axis.set_ylim(-reach * _AXIS_LIMITS[2], reach * _AXIS_LIMITS[1])
  axis.grid(True, zorder=0)
  axis.set_axisbelow(True)
  for spine in ("top", "right"):
    axis.spines[spine].set_visible(False)
  if show_x:
    axis.set_xlabel(text["x"])
  else:
    axis.tick_params(labelbottom=False)
  if show_y:
    axis.set_ylabel(text["z"])
  else:
    axis.tick_params(labelleft=False)


# -- the map ------------------------------------------------------------------


def _set_heading(axis, title: str, subtitle: str, scale: float) -> None:
  """Panel heading in two parts: the quantity, then what it is measured on.

  One line cannot do both without growing wider than the panel. The second line
  is the one a reader needs to interpret the colours -- "Genauigkeit" alone does
  not say *mean over what*, and a caption they have to look away for is a
  caption they do not read.
  """
  axis.set_title(title, pad=14 * scale)
  # Offset in points, not in axes fractions: the same figure is rendered both as
  # one panel of four and on its own, and the gap must not grow with the panel.
  axis.annotate(subtitle, xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0.0, 4.0 * scale), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.6 * scale, color=_MUTED_COLOR)


def _scalar_panel(result: SweepResult, panel: str, language: str) -> tuple:
  """(values, colormap, vmin, vmax, title, subtitle, colorbar label)."""
  text = LABELS[language]
  threshold_mm = result.success_threshold * 1000.0
  if panel == "error":
    return (result.mean_error * 1000.0, "viridis_r", 0.0, None,
            text["error_title"], text["error_subtitle"], text["error"])
  if panel == "success":
    return (result.success_rate * 100.0, "viridis", 0.0, 100.0,
            text["success_title"],
            text["success_subtitle"].format(threshold=threshold_mm),
            text["success_cbar"])
  if panel == "spread":
    return (result.spread * 1000.0, "viridis_r", 0.0, None,
            text["spread_title"], text["spread_subtitle"], text["spread_cbar"])
  raise ValueError(f"Unknown panel {panel!r}; choose from {PANELS}.")


_ARROW_TARGET_COUNTS = (11, 4)
"""Roughly how many arrows to draw along and across the shell.

A fine grid makes a better heat map and a worse arrow field: at 33 x 9 the
arrows overlap into a blue mat. The panel therefore subsamples the same grid to
about this many, which is the density at which a direction is still readable.
"""


def _arrow_stride(result: SweepResult) -> tuple[int, int]:
  along, across = _ARROW_TARGET_COUNTS
  return (
    max(1, len(result.angles) // along),
    max(1, len(result.deviations) // across),
  )


def _draw_deviation_panel(
  axis, result: SweepResult, language: str, scale: float, arrow_scale: float
) -> None:
  """Arrows from every target to the median place the tip came to rest."""
  text = LABELS[language]
  stride_a, stride_r = _arrow_stride(result)
  sub = (slice(None, None, stride_a), slice(None, None, stride_r))
  targets = result.targets_xz[sub]
  delta = (np.median(result.tcp_final, axis=2)[sub] - targets) * arrow_scale
  arrows = axis.quiver(
    targets[..., 0].ravel(), targets[..., 1].ravel(),
    delta[..., 0].ravel(), delta[..., 1].ravel(),
    angles="xy", scale_units="xy", scale=1.0, color=_ARROW_COLOR,
    width=0.006, headwidth=3.6, headlength=4.2, headaxislength=3.8, zorder=5,
  )
  axis.plot(targets[..., 0].ravel(), targets[..., 1].ravel(), linestyle="none",
            marker="o", markersize=2.2 * scale, color=_MUTED_COLOR, zorder=4)
  # A reference arrow rather than a legend entry: the arrows are drawn true to
  # scale in metres, so what the reader needs is the length of a known error.
  # The reference arrow goes in the top-left corner, the one part of a
  # shell-shaped panel that is always empty.
  axis.quiverkey(
    arrows, 0.06, 0.92, _QUIVER_KEY_M * arrow_scale,
    f"{_QUIVER_KEY_M * 1000:.0f} mm", labelpos="E", coordinates="axes",
    color=_ARROW_COLOR, fontproperties={"size": 9 * scale},
  )
  _set_heading(axis, text["deviation_title"], text["deviation_subtitle"], scale)


def draw_map(
  result: SweepResult,
  panels: Sequence[str] = PANELS,
  language: str = "de",
  scale: float = 1.0,
  width: float = 6.3,
  ncols: Optional[int] = None,
  vmax_error: Optional[float] = None,
  arrow_scale: float = 1.0,
):
  """The workspace map: one filled shell per requested panel."""
  from matplotlib.figure import Figure

  from mpl_toolkits.axes_grid1 import make_axes_locatable

  panels = list(panels)
  ncols = ncols or (2 if len(panels) > 2 else len(panels))
  nrows = int(np.ceil(len(panels) / ncols))
  # Height from the panels' own aspect rather than a guess: the axes are
  # equal-aspect, so a figure sized by anything else just adds white space
  # between the rows -- which tight_layout cannot take back out.
  axes_width = max(0.6, width / ncols - _PANEL_DECORATION_W * scale)
  panel_height = axes_width * _AXIS_ASPECT + _PANEL_DECORATION_H * scale
  figure = Figure(figsize=(width, nrows * panel_height + _FIGURE_DECORATION_H * scale))
  axes = figure.subplots(nrows, ncols, squeeze=False)
  corner_x, corner_z = cell_corners(result)

  for index, panel in enumerate(panels):
    row, col = divmod(index, ncols)
    axis = axes[row][col]
    if panel == "deviation":
      _draw_deviation_panel(axis, result, language, scale, arrow_scale)
      # Reserve the colorbar's width anyway, so every panel is drawn at the
      # same scale -- a workspace that is bigger in one panel than in the next
      # is a figure that lies about the geometry.
      spacer = make_axes_locatable(axis).append_axes("right", size="4.5%", pad=0.08)
      spacer.set_axis_off()
    else:
      values, cmap, vmin, vmax, title, subtitle, cbar_label = _scalar_panel(
        result, panel, language
      )
      if panel == "error" and vmax_error is not None:
        vmax = vmax_error
      elif vmax is None:
        vmax = float(np.percentile(values, _COLOUR_PERCENTILE))
      mesh = axis.pcolormesh(corner_x, corner_z, values, cmap=cmap, vmin=vmin,
                             vmax=vmax, shading="flat", zorder=2, rasterized=True)
      _set_heading(axis, title, subtitle, scale)
      cax = make_axes_locatable(axis).append_axes("right", size="4.5%", pad=0.08)
      bar = figure.colorbar(
        mesh, cax=cax, extend="max" if float(values.max()) > vmax else "neither"
      )
      bar.set_label(cbar_label, fontsize=9.5 * scale)
      bar.ax.tick_params(labelsize=9 * scale, length=3, color=_MUTED_COLOR)
      bar.outline.set_visible(False)
    _draw_shell(axis, result.geometry, language, scale)
    _style_workspace_axis(
      axis, result.geometry, language,
      show_x=(row == nrows - 1) or (index + ncols >= len(panels)),
      show_y=(col == 0),
    )

  for index in range(len(panels), nrows * ncols):
    axes[divmod(index, ncols)[0]][divmod(index, ncols)[1]].set_visible(False)

  figure.tight_layout(pad=0.7, h_pad=1.4)
  return figure


# -- the profiles -------------------------------------------------------------


def _draw_angle_profile(axis, result: SweepResult, language: str, scale: float,
                        max_layers: int = 3) -> None:
  """Mean error over the angle, one line per radial layer.

  With more layers than colour slots, show the two edges of the band and its
  centre -- the ones a reader can name; the full grid is in the map.
  """
  text = LABELS[language]
  threshold_mm = result.success_threshold * 1000.0
  n_radius = len(result.deviations)
  if n_radius <= max_layers:
    layer_indices = list(range(n_radius))
  else:
    layer_indices = sorted({0, n_radius // 2, n_radius - 1})
  peak = 0.0
  for slot, index in enumerate(layer_indices):
    deviation_mm = result.deviations[index] * 1000.0
    label = (
      text["layer_centre"] if abs(deviation_mm) < 0.5
      else text["layer"].format(deviation=deviation_mm)
    )
    curve = result.mean_error[:, index] * 1000.0
    peak = max(peak, float(curve.max()))
    axis.plot(
      result.angles, curve,
      color=_LAYER_COLORS[slot % len(_LAYER_COLORS)],
      linestyle=_LAYER_DASHES[slot % len(_LAYER_DASHES)],
      marker=_LAYER_MARKERS[slot % len(_LAYER_MARKERS)],
      markersize=4.4 * scale, markerfacecolor="white", markeredgewidth=1.3,
      markevery=max(1, len(result.angles) // 10),
      linewidth=2.4, solid_capstyle="round", label=label, zorder=3,
    )
  axis.axhline(threshold_mm, color=_THRESHOLD_COLOR, linewidth=1.8,
               linestyle=(0, (4, 2.5)), zorder=2,
               label=text["threshold"].format(threshold=threshold_mm))
  axis.set_xlabel(text["angle"])
  axis.set_ylabel(text["error"])
  axis.set_title(text["profile_angle"], pad=6 * scale)
  axis.set_ylim(0.0, peak * 1.5)  # headroom for the legend
  # Upper left, one column: the error rises toward the middle of the shell in
  # every run so far, so that corner is the one reliably free of data.
  axis.legend(loc="upper left", fontsize=8.6 * scale, handlelength=2.4,
              labelspacing=0.35, borderaxespad=0.3)


def _draw_time_profile(axis, result: SweepResult, language: str, scale: float) -> None:
  """Distance over time, pooled over every rollout, as a median with a band."""
  text = LABELS[language]
  threshold_mm = result.success_threshold * 1000.0
  distance_mm = result.distance.reshape(-1, result.distance.shape[-1]) * 1000.0
  time = np.arange(distance_mm.shape[1]) * result.dt
  low, median, high = np.percentile(distance_mm, [10, 50, 90], axis=0)
  axis.axvspan(time[-1] - result.window, time[-1], color=_GRID_COLOR, alpha=0.6,
               linewidth=0.0, zorder=1, label=text["window"])
  axis.fill_between(time, low, high, color=_LAYER_COLORS[0], alpha=0.20,
                    linewidth=0.0, zorder=2, label=text["band"])
  axis.plot(time, median, color=_LAYER_COLORS[0], linewidth=2.6,
            solid_capstyle="round", zorder=3, label=text["median"])
  axis.axhline(threshold_mm, color=_THRESHOLD_COLOR, linewidth=1.8,
               linestyle=(0, (4, 2.5)), zorder=4,
               label=text["threshold"].format(threshold=threshold_mm))
  axis.set_xlabel(text["time"])
  axis.set_ylabel(text["distance"])
  axis.set_title(text["profile_time"], pad=6 * scale)
  axis.set_xlim(0.0, time[-1])
  axis.set_ylim(0.0, float(high.max()) * 1.05)
  axis.legend(loc="upper right", fontsize=8.6 * scale, handlelength=2.4)


_PROFILE_PANELS = {"angle": _draw_angle_profile, "time": _draw_time_profile}


def draw_profiles(
  result: SweepResult,
  panels: Sequence[str] = PROFILE_PANELS,
  language: str = "de",
  scale: float = 1.0,
  width: float = 6.3,
):
  """The one-dimensional cuts: error along the shell, and approach over time."""
  from matplotlib.figure import Figure

  panels = list(panels)
  # Line plots have no natural aspect, so the height is a reading decision: a
  # panel standing alone gets a taller box than one of a pair.
  figure = Figure(figsize=(width, width * (0.45 if len(panels) > 1 else 0.62)))
  axes = figure.subplots(1, len(panels), squeeze=False)[0]

  for axis, panel in zip(axes, panels, strict=True):
    if panel not in _PROFILE_PANELS:
      raise ValueError(f"Unknown profile panel {panel!r}; choose from {PROFILE_PANELS}.")
    _PROFILE_PANELS[panel](axis, result, language, scale)
    axis.grid(True, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
      axis.spines[spine].set_visible(False)

  figure.tight_layout(pad=0.7)
  return figure


# -- saving -------------------------------------------------------------------


def save_figures(
  result: SweepResult,
  out_base: Path,
  figures: Sequence[str] = FIGURES,
  panels: Sequence[str] = PANELS,
  language: str = "de",
  scale: float = 1.0,
  width: float = 6.3,
  formats: Sequence[str] = ("pdf", "png"),
  dpi: int = 300,
  vmax_error: Optional[float] = None,
  arrow_scale: float = 1.0,
  single_panels: bool = True,
  single_width: Optional[float] = None,
) -> list[Path]:
  """Render the requested figures next to their sweep file.

  Every panel is written **twice**: once in its combined figure, and once on its
  own as ``<base>_<figure>_<panel>.*``, so a page that wants one map large does
  not have to be given all four small. Same reason ``target_figure`` always
  writes its ``_ohne_bahn`` twin -- which crop a layout wants is decided long
  after the numbers were measured, and rendering both costs a second.

  A single panel is drawn at ``single_width`` (by default the same ``width``),
  because the type sizes assume the figure is printed at about its own width:
  designing a lone panel small and then blowing it up to the text width would
  magnify the labels with it.

  The publication style is applied inside an ``rc_context`` and never globally,
  for the same reason ``target_figure.save_figure`` does it: a caller that has
  its own figures open must not have them restyled behind its back.
  """
  import matplotlib as mpl

  out_base = Path(out_base)
  out_base.parent.mkdir(parents=True, exist_ok=True)
  single_width = single_width or width
  written: list[Path] = []

  def build(name: str, subset: Optional[Sequence[str]], figure_width: float):
    if name == "karte":
      return draw_map(result, panels=subset or panels, language=language, scale=scale,
                      width=figure_width, vmax_error=vmax_error, arrow_scale=arrow_scale)
    if name == "profil":
      return draw_profiles(result, panels=subset or PROFILE_PANELS, language=language,
                           scale=scale, width=figure_width)
    raise ValueError(f"Unknown figure {name!r}; choose from {FIGURES}.")

  def save(figure, stem: str) -> None:
    for suffix in formats:
      path = out_base.with_name(f"{stem}.{suffix}")
      figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
      written.append(path)
      print(f"[figure] {path}")

  with mpl.rc_context():
    apply_style(scale)
    for name in figures:
      save(build(name, None, width), f"{out_base.name}_{name}")
      if not single_panels:
        continue
      subsets = panels if name == "karte" else PROFILE_PANELS
      for panel in subsets:
        save(build(name, [panel], single_width), f"{out_base.name}_{name}_{panel}")
  return written


def main(argv: Optional[list[str]] = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--sweep", required=True, help="the .npz written by workspace_sweep")
  parser.add_argument("--figures", default=",".join(FIGURES),
                      help=f"comma-separated subset of {FIGURES}")
  parser.add_argument("--panels", default=",".join(PANELS),
                      help=f"map panels, comma-separated subset of {PANELS}")
  parser.add_argument("--language", default=_default_language(), choices=sorted(LABELS))
  parser.add_argument("--scale", type=float, default=1.0, help="type-size multiplier")
  parser.add_argument("--width", type=float, default=6.3,
                      help="figure width [inch]; type is sized for printing at it")
  parser.add_argument("--threshold", type=float, default=None,
                      help="success threshold [mm]; default is the task's own")
  parser.add_argument("--vmax-error", type=float, default=None,
                      help="upper end of the error colour scale [mm], to compare sweeps")
  parser.add_argument("--arrow-scale", type=float, default=1.0,
                      help="exaggeration of the deviation arrows (1 = true to scale)")
  parser.add_argument("--no-single", action="store_true",
                      help="skip the one-file-per-panel renderings")
  parser.add_argument("--single-width", type=float, default=None,
                      help="width [inch] of a single-panel figure (default: --width)")
  parser.add_argument("--out", default=None, help="output base path (default: next to --sweep)")
  parser.add_argument("--summary", action="store_true", help="also print the sweep summary")
  args = parser.parse_args(argv)

  sweep_path = Path(args.sweep)
  result = SweepResult.load(sweep_path)
  if args.threshold is not None:
    result = dataclasses.replace(result, success_threshold=args.threshold / 1000.0)
  if args.summary:
    from .workspace_sweep import summary_text

    print(summary_text(result))
  save_figures(
    result,
    out_base=Path(args.out) if args.out else sweep_path.with_suffix(""),
    figures=[name.strip() for name in args.figures.split(",") if name.strip()],
    panels=[name.strip() for name in args.panels.split(",") if name.strip()],
    language=args.language,
    scale=args.scale,
    width=args.width,
    vmax_error=args.vmax_error,
    arrow_scale=args.arrow_scale,
    single_panels=not args.no_single,
    single_width=args.single_width,
  )


if __name__ == "__main__":
  main()
