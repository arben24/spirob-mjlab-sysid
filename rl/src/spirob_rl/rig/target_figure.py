"""Publication figure of the SpiRob workspace: soll, ist, and the arm between them.

A photograph of ``target_gui.py`` is a picture of a *tool*: dark chrome, hairlines
tuned for a screen, fonts that turn to mush at print size. What a thesis needs is
the same *numbers* drawn again for paper -- vector, white, thick, labelled, with
the deviation as a figure in millimetres rather than a status line.

So this module never touches pixels of the GUI. It takes the raw state the GUI
already holds -- the commanded target, the measured TCP, the arm's segment points
from forward kinematics, and the recent path of the tip -- and redraws it with
matplotlib. The same style as ``inverted_pendulum/sysid_figure.py``: Okabe-Ito
colours, a second non-colour identity (open vs filled, solid vs dashed) for every
series so the figure survives greyscale, and PDF as the primary output.

Three ways in, all producing the same ``Scene``:

* the **button** in ``target_gui.py`` ("Abbildung speichern"), which is the usual
  one -- it captures what is on screen at that moment;
* ``--listen``, which records straight from the bridge's telemetry for a few
  seconds without any GUI running;
* ``--scene``, which re-renders a previously captured ``*.json``. Every capture
  writes that file next to the figure, so a figure can be re-drawn later in
  another language, size or without its trace, and never has to be re-measured.

Every export writes **two** renderings whenever there is a tip path: with it,
and ``*_ohne_bahn`` without. Which one a page wants is a layout question decided
long after the rig has moved on, and rendering both costs a second.

*Which part* of the path is shown is a third question, and equally not one to
answer while the arm is still moving: the scene stores a timestamp per path
point, ``Scene.cropped`` trims to a window, and :mod:`spirob_rl.rig.target_crop` is the
GUI for choosing that window by dragging. The choice lands in ``*_crop.json``
next to the scene and is picked up automatically on every later render.

Usage::

    uv run python -m spirob_rl.rig.target_figure --listen --duration 8
    uv run python -m spirob_rl.rig.target_figure --scene build/rl/figures/reach_1.json \\
        --language en --width 5.2 --trace-start 2.4 --trace-end 7.1
    uv run python -m spirob_rl.rig.target_figure --target 0.20 0.37   # geometry only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from spirob.paths import build_dir

from . import telemetry, workspace

BUILD_DIR = build_dir("rl", "figures")


def _default_language() -> str:
  """Figure language, following the repository's one localisation switch.

  ``SPIROB_FIG_LOCALE=de`` renders the German variants used in the thesis (see
  ``spirob.plotstyle``); everything else is English, like every other figure in
  this repository. ``--language`` still overrides it explicitly.
  """
  import os

  return "de" if os.environ.get("SPIROB_FIG_LOCALE", "").lower().startswith("de") else "en"


# Categorical slots of the reference palette, in their fixed order -- the same
# three ``sysid_figure`` uses, so the two figure families of the thesis share one
# colour language. Order is the colour-blindness safety mechanism and is not
# rearranged by taste; print and greyscale are covered by the marker shapes.
_ARM = "#0071b2"        # the robot itself, and its tip: what was measured
_TARGET = "#d55c00"     # the commanded target: what was asked for
_SHELL_EDGE = "#009e74"  # the trained workspace
_SHELL_FILL = "#e9f4ee"
_TRACE = "#8c3d8f"
"""The tip's path: dark purple.

Was a light blue, which is unreadable for the one thing it has to survive --
being drawn *on top of* the shell's light green fill, at hairline width. Purple
is the remaining hue that is neither the arm's blue nor the shell's green nor
the target's orange, and dark enough to hold its own against the fill.
"""
_BASE_COLOR = "#3a3a38"
_GRID_COLOR = "#d8d8d4"
_TEXT_COLOR = "#0b0b0b"
_MUTED_COLOR = "#52514e"

_SHELL_SAMPLES = 241
_MARGIN_M = 0.05

LABELS = {
  "de": {
    "x": "x [m]",
    "z": "z [m]",
    "shell": "erreichbare Schale (±{band:g} m)",
    "angle_limit": "Winkelgrenze ±{limit:.1f} rad",
    "arm": "Arm (gemessene Pose)",
    "arm_rest": "Arm (Ruhelage, keine Messung)",
    "target": "Ziel (soll)",
    "actual": "TCP (ist)",
    "trace": "Bahn der Spitze ({seconds:.0f} s)",
    "trace_start": "Bahnanfang",
    "base": "Basis",
    "soll": "soll",
    "ist": "ist",
    "deviation": "Abweichung",
    "no_measurement": "keine Ist-Position (kein Winkelbrett)",
  },
  "en": {
    "x": "x [m]",
    "z": "z [m]",
    "shell": "reachable shell (±{band:g} m)",
    "angle_limit": "angle limit ±{limit:.1f} rad",
    "arm": "arm (measured pose)",
    "arm_rest": "arm (rest pose, not measured)",
    "target": "target (commanded)",
    "actual": "TCP (measured)",
    "trace": "tip path ({seconds:.0f} s)",
    "trace_start": "path start",
    "base": "base",
    "soll": "cmd",
    "ist": "meas",
    "deviation": "deviation",
    "no_measurement": "no measured position (no angle board)",
  },
}


@dataclass(frozen=True)
class Scene:
  """One moment of the rig, as numbers -- everything a figure of it needs.

  Deliberately plain floats and lists: this is what gets written next to the
  figure as JSON, and re-reading it must not require the bridge, the task
  registry or a MuJoCo model. A figure can therefore be re-drawn months later
  from the file alone.
  """

  target: tuple[float, float]
  """Commanded TCP target (x, z) [m] in the robot base frame."""
  tcp_actual: Optional[tuple[float, float]] = None
  """Measured TCP (x, z) [m], or None when nothing on the rig knew."""
  arm: tuple[tuple[float, float], ...] = ()
  """Segment origins base->tip from forward kinematics, ending at the TCP."""
  trace: tuple[tuple[float, float], ...] = ()
  """Recent measured tip positions, oldest first."""
  trace_t: tuple[float, ...] = ()
  """Time of each trace point [s], relative to the first one.

  Stored so the path can be trimmed *after* the fact -- "from second 2.4 to
  second 7.1 of this reach" is a decision nobody can make while the arm is still
  moving. See :meth:`cropped` and :mod:`spirob_rl.rig.target_crop`.
  """
  trace_seconds: float = 0.0
  """Wall-clock span the trace covers [s] -- goes into its legend entry."""
  from_rig: bool = False
  """True when this was captured from a running bridge.

  Distinguishes "the rig had nothing to report" from "no rig was involved": only
  the former deserves the figure's note about the missing angle board. A purely
  geometric figure of the workspace is not missing a measurement.
  """
  task: str = ""
  captured_at: str = ""
  note: str = ""

  @property
  def deviation_m(self) -> Optional[float]:
    if self.tcp_actual is None:
      return None
    return math.hypot(self.tcp_actual[0] - self.target[0], self.tcp_actual[1] - self.target[1])

  @property
  def trace_times(self) -> tuple[float, ...]:
    """Trace timestamps, spaced evenly if the scene predates ``trace_t``.

    A scene captured before timestamps existed still has a path and a duration;
    assuming a constant rate over it is wrong in detail but right in scale, and
    it keeps every stored scene croppable. Anything captured since carries the
    real times.
    """
    if len(self.trace_t) == len(self.trace):
      return self.trace_t
    if len(self.trace) < 2:
      return (0.0,) * len(self.trace)
    step = self.trace_seconds / (len(self.trace) - 1)
    return tuple(i * step for i in range(len(self.trace)))

  def cropped(self, t_start: float, t_end: float) -> "Scene":
    """The same scene with the tip path trimmed to ``[t_start, t_end]``.

    Only the path is cut. Target, measured tip and arm pose describe the moment
    of capture, not the window -- trimming a path does not move the arm.
    """
    times = self.trace_times
    keep = [i for i, t in enumerate(times) if t_start <= t <= t_end]
    if not keep:
      return replace(self, trace=(), trace_t=(), trace_seconds=0.0)
    first = times[keep[0]]
    return replace(
      self,
      trace=tuple(self.trace[i] for i in keep),
      trace_t=tuple(times[i] - first for i in keep),
      trace_seconds=times[keep[-1]] - first,
    )

  def to_dict(self) -> dict:
    return {
      "target": list(self.target),
      "tcp_actual": list(self.tcp_actual) if self.tcp_actual is not None else None,
      "arm": [list(p) for p in self.arm],
      "trace": [list(p) for p in self.trace],
      "trace_t": [round(t, 4) for t in self.trace_t],
      "trace_seconds": self.trace_seconds,
      "from_rig": self.from_rig,
      "task": self.task,
      "captured_at": self.captured_at,
      "note": self.note,
    }

  @classmethod
  def from_dict(cls, payload: dict) -> "Scene":
    actual = payload.get("tcp_actual")
    return cls(
      target=(float(payload["target"][0]), float(payload["target"][1])),
      tcp_actual=(float(actual[0]), float(actual[1])) if actual else None,
      arm=tuple((float(x), float(z)) for x, z in payload.get("arm", [])),
      trace=tuple((float(x), float(z)) for x, z in payload.get("trace", [])),
      trace_t=tuple(float(t) for t in payload.get("trace_t", [])),
      trace_seconds=float(payload.get("trace_seconds", 0.0)),
      from_rig=bool(payload.get("from_rig", False)),
      task=str(payload.get("task", "")),
      captured_at=str(payload.get("captured_at", "")),
      note=str(payload.get("note", "")),
    )

  def save(self, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(self.to_dict(), indent=2))
    return path

  @classmethod
  def load(cls, path: Path) -> "Scene":
    return cls.from_dict(json.loads(Path(path).read_text()))


def apply_style(scale: float = 1.0) -> None:
  """Typography and chrome for a figure that will be scaled into a page.

  Sizes assume inclusion at roughly the figure's own width: 10-11 pt text
  survives that without the shrink-to-fit blur that makes a screenshot look
  pasted in.
  """
  import matplotlib as mpl

  mpl.rcParams.update({
    "font.size": 11 * scale,
    "axes.labelsize": 11.5 * scale,
    "axes.titlesize": 12 * scale,
    "legend.fontsize": 10 * scale,
    "xtick.labelsize": 10 * scale,
    "ytick.labelsize": 10 * scale,
    "axes.edgecolor": _GRID_COLOR,
    "axes.labelcolor": _TEXT_COLOR,
    "text.color": _TEXT_COLOR,
    "xtick.color": _MUTED_COLOR,
    "ytick.color": _MUTED_COLOR,
    "axes.linewidth": 0.8,
    "grid.color": _GRID_COLOR,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",  # a dashed grid reads as a threshold, not chrome
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,  # embed TrueType, so the PDF stays editable/searchable
    "ps.fonttype": 42,
  })


def _shell_polygons() -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
  """(outer, centre, inner) point lists of the shell band, in world metres."""
  outer, centre, inner = [], [], []
  for i in range(_SHELL_SAMPLES):
    angle = -workspace.ANGLE_LIMIT + i * (2 * workspace.ANGLE_LIMIT / (_SHELL_SAMPLES - 1))
    s, c = math.sin(angle), math.cos(angle)
    r = workspace.shell_radius(angle)
    outer.append(((r + workspace.SHELL_BAND) * s, (r + workspace.SHELL_BAND) * c))
    centre.append((r * s, r * c))
    inner.append(((r - workspace.SHELL_BAND) * s, (r - workspace.SHELL_BAND) * c))
  return outer, centre, inner


def draw_workspace(axis, language: str = "de", scale: float = 1.0) -> dict:
  """The parts that never change: shell band, angle limits, base marker.

  Its own function because the crop GUI draws the same background live while you
  drag -- what you choose the window on is then literally the figure you get,
  and the two cannot drift apart. Returns the legend entries it created.
  """
  from matplotlib.patches import Polygon

  text = LABELS[language]
  outer, centre, inner = _shell_polygons()
  entries: dict[str, object] = {}

  band = Polygon(
    outer + inner[::-1], closed=True, facecolor=_SHELL_FILL, edgecolor=_SHELL_EDGE,
    linewidth=1.2, zorder=1,
    # The shell's parabola belongs in the caption, not in a legend entry that has
    # to stay readable at half column width.
    label=text["shell"].format(band=workspace.SHELL_BAND),
  )
  axis.add_patch(band)
  entries["shell"] = band
  axis.plot(
    [p[0] for p in centre], [p[1] for p in centre],
    color=_SHELL_EDGE, linewidth=1.8, linestyle=(0, (6, 3)), zorder=2,
  )

  # The two rays the sampling is clipped at. Drawn faint: they bound the figure's
  # argument but are not part of it.
  for sign in (-1.0, 1.0):
    x, z = workspace.shell_point(sign * workspace.ANGLE_LIMIT)
    axis.plot([0.0, x * 1.06], [0.0, z * 1.06], color=_MUTED_COLOR, linewidth=0.8,
              linestyle=(0, (2, 4)), zorder=2)
  # Along the ray rather than at its end: the end is where the base label and the
  # x-axis already are, and a limit is easiest to read on the line it belongs to.
  x_lim, z_lim = workspace.shell_point(workspace.ANGLE_LIMIT)
  axis.annotate(
    text["angle_limit"].format(limit=workspace.ANGLE_LIMIT),
    xy=(x_lim * 0.78, z_lim * 0.78), xytext=(0, -6 * scale), textcoords="offset points",
    color=_MUTED_COLOR, fontsize=9 * scale, ha="center", va="top",
    rotation=math.degrees(math.atan2(math.cos(workspace.ANGLE_LIMIT),
                                     math.sin(workspace.ANGLE_LIMIT))),
    rotation_mode="anchor",
  )

  axis.plot([0.0], [0.0], linestyle="none", marker="s", markersize=8 * scale,
            color=_BASE_COLOR, zorder=7)
  # Left of the base: the +x side below the arm is where the angle-limit ray and
  # its label run.
  axis.annotate(text["base"], xy=(0.0, 0.0), xytext=(-8 * scale, -12 * scale),
                textcoords="offset points", ha="right", color=_BASE_COLOR,
                fontsize=9.5 * scale)
  return entries


def draw_measurement(
  axis,
  scene: Scene,
  language: str = "de",
  scale: float = 1.0,
  show_trace: bool = True,
  annotate: bool = True,
) -> dict:
  """Everything the rig contributed: path, arm pose, target, measured tip.

  Separate from :func:`draw_workspace` so the crop GUI can redraw only this part
  while the background stays put. Returns the legend entries it created.
  """
  text = LABELS[language]
  entries: dict[str, object] = {}

  if show_trace and len(scene.trace) > 1:
    entries["trace"] = axis.plot(
      [p[0] for p in scene.trace], [p[1] for p in scene.trace],
      color=_TRACE, linewidth=2.0, solid_capstyle="round", zorder=3,
      label=text["trace"].format(seconds=scene.trace_seconds),
    )[0]
    # Which end of the path is the beginning is not visible in a line, and for a
    # reach it is the whole story. Open diamond in the path's own colour: same
    # family as the line, no collision with the target's plus or the tip's disc.
    entries["trace_start"] = axis.plot(
      [scene.trace[0][0]], [scene.trace[0][1]], linestyle="none", marker="D",
      markersize=7 * scale, markerfacecolor="white", markeredgecolor=_TRACE,
      markeredgewidth=1.8, zorder=6, label=text["trace_start"],
    )[0]

  if scene.arm:
    entries["arm"] = axis.plot(
      [p[0] for p in scene.arm], [p[1] for p in scene.arm],
      color=_ARM, linewidth=3.4, solid_capstyle="round", solid_joinstyle="round",
      zorder=4, label=text["arm"],
    )[0]
    axis.plot(
      [p[0] for p in scene.arm[1:-1]], [p[1] for p in scene.arm[1:-1]],
      linestyle="none", marker="o", markersize=4.2 * scale, markerfacecolor="white",
      markeredgecolor=_ARM, markeredgewidth=1.1, zorder=5,
    )

  # -- soll, ist, and the gap --------------------------------------------------
  if scene.tcp_actual is not None and annotate:
    axis.plot(
      [scene.target[0], scene.tcp_actual[0]], [scene.target[1], scene.tcp_actual[1]],
      color=_MUTED_COLOR, linewidth=1.2, linestyle=(0, (3, 2)), zorder=6,
    )

  # Open marker for the wish, filled for the measurement -- the difference has to
  # survive greyscale, so it is a shape difference and not only a colour one.
  entries["target"] = axis.plot(
    [scene.target[0]], [scene.target[1]], linestyle="none", marker="P",
    markersize=11 * scale, markerfacecolor="white", markeredgecolor=_TARGET,
    markeredgewidth=2.0, zorder=8, label=text["target"],
  )[0]
  if scene.tcp_actual is not None:
    entries["actual"] = axis.plot(
      [scene.tcp_actual[0]], [scene.tcp_actual[1]], linestyle="none", marker="o",
      markersize=8.5 * scale, markerfacecolor=_ARM, markeredgecolor="white",
      markeredgewidth=1.4, zorder=9, label=text["actual"],
    )[0]

  if annotate:
    _annotate_points(axis, scene, text, scale)
  return entries


def autoscale(axis, scene: Scene, show_trace: bool = True) -> None:
  """Frame the shell plus whatever the scene adds, with equal aspect.

  Takes the *full* scene even when a cropped path is drawn: a view that jumps
  while the crop window is dragged makes the two look like different reaches.
  """
  outer, _centre, inner = _shell_polygons()
  points = [*outer, *inner, scene.target, *scene.arm]
  if scene.tcp_actual is not None:
    points.append(scene.tcp_actual)
  if show_trace:
    points.extend(scene.trace)
  xs = [p[0] for p in points]
  zs = [p[1] for p in points]
  axis.set_xlim(min(xs) - _MARGIN_M, max(xs) + _MARGIN_M)
  axis.set_ylim(min(zs) - _MARGIN_M, max(zs) + _MARGIN_M)
  axis.set_aspect("equal", adjustable="box")  # metres are metres in both directions


def draw_scene(
  scene: Scene,
  language: str = "de",
  width: float = 6.9,
  height: Optional[float] = None,
  scale: float = 1.0,
  show_trace: bool = True,
  annotate: bool = True,
  title: str = "",
  legend: bool = True,
):
  """Draw one scene and return ``(figure, axis)``.

  Split from :func:`save_figure` so a caller assembling a multi-panel figure of
  several reaches can place these axes itself.

  Built without pyplot on purpose: pyplot owns a *global* backend and figure
  registry, and switching that from inside a Tk callback or a running crop GUI
  closes every open window -- including the one that asked for the export. A
  detached ``Figure`` with its own Agg canvas has no such reach.
  """
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from matplotlib.figure import Figure

  apply_style(scale)
  text = LABELS[language]
  figure = Figure(figsize=(width, height or width * 0.78))
  FigureCanvasAgg(figure)  # savefig swaps in the right canvas per format
  axis = figure.add_subplot(1, 1, 1)

  # Legend entries are collected by role, not in drawing order: drawing runs
  # background-to-foreground, while the legend should read measurement-first.
  entries = draw_workspace(axis, language=language, scale=scale)
  entries.update(draw_measurement(
    axis, scene, language=language, scale=scale, show_trace=show_trace, annotate=annotate
  ))

  autoscale(axis, scene, show_trace=show_trace)
  axis.set_xlabel(text["x"])
  axis.set_ylabel(text["z"])
  axis.grid(True, alpha=0.9)
  axis.set_axisbelow(True)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)
  if title:
    axis.set_title(title)

  if legend:
    handles = [entries[key] for key in
               ("arm", "actual", "target", "trace", "trace_start", "shell") if key in entries]
    axis.legend(
      handles=handles, labels=[h.get_label() for h in handles],
      loc="upper center", bbox_to_anchor=(0.5, -0.14),
      ncol=2 if len(handles) > 3 else len(handles) or 1,
      borderaxespad=0.0, handlelength=2.4, columnspacing=1.6,
    )
  return figure, axis


def _annotate_points(axis, scene: Scene, text: dict, scale: float) -> None:
  """The numbers, as one aligned block in the free upper-left corner.

  Not as labels next to the markers, which is what this started as: in a *good*
  run soll and ist are millimetres apart, so exactly when the figure matters
  most, three labels pile onto the same square centimetre. The corner above the
  shell's left flank is empty for every pose the arm can take, so the block
  always lands in white space, and its lines are colour-coded to the markers
  they describe -- monospaced so the coordinates line up under each other.
  """
  lines = [(f"{text['soll']:<5s} ({scene.target[0]:+.3f}, {scene.target[1]:+.3f}) m", _TARGET)]
  if scene.tcp_actual is None:
    if scene.from_rig:  # a rig that could not say where it is; not a plain sketch
      lines.append((text["no_measurement"], _MUTED_COLOR))
  else:
    lines.append(
      (f"{text['ist']:<5s} ({scene.tcp_actual[0]:+.3f}, {scene.tcp_actual[1]:+.3f}) m", _ARM)
    )
    lines.append(
      (f"{text['deviation']} {(scene.deviation_m or 0.0) * 1000:.1f} mm", _MUTED_COLOR)
    )

  for row, (line, color) in enumerate(lines):
    axis.annotate(
      line, xy=(0.0, 1.0), xycoords="axes fraction",
      xytext=(8 * scale, -(6 + row * 14) * scale), textcoords="offset points",
      ha="left", va="top", color=color, fontsize=9.5 * scale, family="monospace",
    )


NO_TRACE_SUFFIX = "_ohne_bahn"
"""Filename suffix of the second, path-free rendering of the same scene.

Both are always written: which one a page wants is a layout decision made much
later, and re-running an export because the wrong variant was produced means
going back to a figure the rig has long since moved past.
"""


def save_figure(
  scene: Scene,
  out_base: Optional[Path] = None,
  formats: Sequence[str] = ("pdf", "png"),
  dpi: int = 300,
  write_scene: bool = True,
  also_without_trace: bool = True,
  **draw_kwargs,
) -> list[Path]:
  """Draw the scene and write it in every requested format, plus its raw numbers.

  Two renderings whenever there is a path to show: ``<base>.*`` with it and
  ``<base>_ohne_bahn.*`` without. The ``.json`` alongside is the point of the
  whole module -- a figure that can be re-rendered is a figure that never has to
  be re-measured on the rig.
  """
  import matplotlib as mpl

  out_base = out_base or default_out_base()
  out_base.parent.mkdir(parents=True, exist_ok=True)

  variants = [(out_base, draw_kwargs)]
  if also_without_trace and len(scene.trace) > 1 and draw_kwargs.get("show_trace", True):
    variants.append((
      out_base.with_name(out_base.name + NO_TRACE_SUFFIX), {**draw_kwargs, "show_trace": False},
    ))

  written: list[Path] = []
  # The publication style stays inside this call: some of it (fonttype, savefig
  # facecolor) is only read at save time, and a caller that has its own figures
  # open -- the crop GUI does -- must not have them restyled behind its back.
  with mpl.rc_context():
    for base, kwargs in variants:
      figure, _axis = draw_scene(scene, **kwargs)
      for suffix in formats:
        path = base.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
        print(f"[figure] {path}")

  if write_scene:
    scene_path = scene.save(out_base.with_suffix(".json"))
    print(f"[figure] {scene_path}  (Rohdaten, erneut renderbar mit --scene)")
  return written


# -- cropping the tip path -----------------------------------------------------


def crop_file_path(scene_path: Path) -> Path:
  """Where the chosen path window for ``scene_path`` lives."""
  scene_path = Path(scene_path)
  return scene_path.with_name(f"{scene_path.stem}_crop.json")


def save_crop_file(scene_path: Path, t_start: float, t_end: float) -> Path:
  """Store a path window next to its scene.

  A separate file rather than a trimmed scene: the recording stays whole, so a
  crop can be widened again later. Same arrangement as
  ``inverted_pendulum/sysid_crop.py``.
  """
  path = crop_file_path(scene_path)
  path.write_text(json.dumps({"t_start": round(t_start, 4), "t_end": round(t_end, 4)}, indent=2))
  return path


def load_crop_file(scene_path: Path) -> Optional[tuple[float, float]]:
  """The stored path window for a scene, or None if there is none."""
  path = crop_file_path(scene_path)
  if not path.exists():
    return None
  try:
    payload = json.loads(path.read_text())
    return float(payload["t_start"]), float(payload["t_end"])
  except (ValueError, KeyError, TypeError, OSError):
    print(f"[figure] {path.name} ist unlesbar und wird ignoriert")
    return None


def default_out_base(prefix: str = "workspace") -> Path:
  return BUILD_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


# -- capture without the GUI ---------------------------------------------------


def capture_from_telemetry(
  duration_s: float = 5.0,
  port: int = telemetry.TELEMETRY_UDP_PORT,
  poll_s: float = 0.02,
) -> Scene:
  """Listen to the bridge for a while and build a scene from what arrived.

  The trace is every measured tip position over the window, the rest is the last
  tick -- so a short ``--duration`` gives a snapshot and a long one gives the
  whole reach.
  """
  from . import kinematics

  try:
    geometry = kinematics.load_chain_geometry()
  except Exception as exc:  # a tip dot without an arm is still a figure
    geometry = None
    print(f"[figure] Kettengeometrie nicht ladbar, Arm wird weggelassen: {exc}")

  receiver = telemetry.TelemetryReceiver(port=port)
  print(f"[figure] hoere {duration_s:g} s auf udp://127.0.0.1:{port} ...")
  trace: list[tuple[float, float]] = []
  trace_t: list[float] = []
  last: Optional[telemetry.BridgeState] = None
  deadline = time.monotonic() + duration_s
  try:
    while time.monotonic() < deadline:
      for state in receiver.drain():
        last = state
        if state.tcp_actual is not None:
          trace.append(state.tcp_actual)
          trace_t.append(state.t)  # bridge clock; made relative below
      time.sleep(poll_s)
  finally:
    receiver.close()

  if last is None:
    raise SystemExit(
      f"Keine Telemetrie auf Port {port} -- laeuft die Bridge? "
      "(oder --target x z fuer eine reine Geometrie-Abbildung)"
    )
  arm: tuple[tuple[float, float], ...] = ()
  q = last.signals.get("joint_pos")
  if geometry is not None and q is not None and len(q) == geometry.num_joints:
    arm = tuple((float(x), float(z)) for x, z in kinematics.forward_kinematics(q, geometry))
  relative = tuple(t - trace_t[0] for t in trace_t) if trace_t else ()
  return Scene(
    target=last.target,
    tcp_actual=last.tcp_actual,
    arm=arm,
    trace=tuple(trace),
    trace_t=relative,
    trace_seconds=relative[-1] if relative else 0.0,
    from_rig=True,
    task=last.task,
    captured_at=datetime.now().isoformat(timespec="seconds"),
    note=f"{len(trace)} Messpunkte, tick {last.tick}",
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  source = parser.add_argument_group("Quelle (eine davon)")
  source.add_argument("--scene", type=Path, default=None,
                      help="Zuvor aufgenommene Szene (*.json) erneut rendern")
  source.add_argument("--listen", action="store_true",
                      help="Direkt aus der Bridge-Telemetrie aufnehmen")
  source.add_argument("--target", type=float, nargs=2, metavar=("X", "Z"), default=None,
                      help="Nur Geometrie: Schale und dieses Ziel, ohne Hardware")

  parser.add_argument("--duration", type=float, default=5.0,
                      help="Aufnahmefenster fuer --listen [s]; die Bahn deckt es ab")
  parser.add_argument("--telemetry-port", type=int, default=telemetry.TELEMETRY_UDP_PORT)
  parser.add_argument("--out", type=Path, default=None, help="Ausgabepfad ohne Endung")
  parser.add_argument("--language", choices=("de", "en"), default=_default_language())
  parser.add_argument("--width", type=float, default=6.9,
                      help="Breite [inch] (6.9 ~ 17.5 cm Textbreite)")
  parser.add_argument("--height", type=float, default=None, help="Hoehe [inch]")
  parser.add_argument("--scale", type=float, default=1.0,
                      help="Schriftskalierung fuer halbbreite Platzierung")
  parser.add_argument("--format", default="pdf,png", help="Kommaliste: pdf, png, svg, eps")
  parser.add_argument("--dpi", type=int, default=300)
  parser.add_argument("--title", default="", help="Titel (im Papier meist die Bildunterschrift)")
  parser.add_argument("--no-trace", dest="trace", action="store_false",
                      help="Bahn der Spitze weglassen (die Variante *_ohne_bahn entsteht sonst "
                           "ohnehin zusaetzlich)")
  parser.add_argument("--trace-start", type=float, default=None,
                      help="Bahn erst ab dieser Sekunde der Aufnahme zeichnen")
  parser.add_argument("--trace-end", type=float, default=None,
                      help="Bahn nur bis zu dieser Sekunde zeichnen")
  parser.add_argument("--ignore-crop-file", action="store_true",
                      help="Ein neben der Szene liegendes *_crop.json nicht beruecksichtigen")
  parser.add_argument("--no-annotate", dest="annotate", action="store_false",
                      help="Zahlen an den Markern weglassen")
  parser.add_argument("--no-legend", dest="legend", action="store_false")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  window: Optional[tuple[float, float]] = None
  if args.scene is not None:
    scene = Scene.load(args.scene)
    # Re-rendering writes over the figures belonging to that scene: same numbers,
    # new options, so a stack of near-identical files never accumulates.
    default_base = args.scene.with_suffix("")
    if not args.ignore_crop_file:
      window = load_crop_file(args.scene)
      if window is not None:
        print(f"[figure] {crop_file_path(args.scene).name}: Bahn {window[0]:.2f} - {window[1]:.2f} s")
  elif args.target is not None:
    scene = Scene(
      target=(args.target[0], args.target[1]),
      captured_at=datetime.now().isoformat(timespec="seconds"),
      note="nur Geometrie",
    )
    default_base = default_out_base("workspace")
  else:
    scene = capture_from_telemetry(args.duration, port=args.telemetry_port)
    default_base = default_out_base("reach")

  # Explicit flags beat the stored window, the stored window beats the full path.
  if args.trace_start is not None or args.trace_end is not None:
    times = scene.trace_times
    window = (
      args.trace_start if args.trace_start is not None else (times[0] if times else 0.0),
      args.trace_end if args.trace_end is not None else (times[-1] if times else 0.0),
    )
  if window is not None and scene.trace:
    scene = scene.cropped(*window)
    print(f"[figure] Bahn zugeschnitten: {len(scene.trace)} Punkte, {scene.trace_seconds:.2f} s")

  info = workspace.describe(*scene.target)
  print(
    f"[figure] soll {scene.target[0]:+.3f} {scene.target[1]:+.3f} "
    f"(winkel {info.angle:+.2f} rad, abw zur Schale {info.deviation:+.3f} m, "
    f"{'auf der Schale' if info.ok else 'ausserhalb'})"
  )
  if scene.deviation_m is not None:
    print(f"[figure] ist  {scene.tcp_actual[0]:+.3f} {scene.tcp_actual[1]:+.3f}   "
          f"abweichung {scene.deviation_m * 1000:.1f} mm")

  save_figure(
    scene,
    out_base=args.out or default_base,
    formats=[f.strip() for f in args.format.split(",") if f.strip()],
    dpi=args.dpi,
    write_scene=args.scene is None,  # re-rendering must not overwrite its source
    language=args.language,
    width=args.width,
    height=args.height,
    scale=args.scale,
    show_trace=args.trace,
    annotate=args.annotate,
    title=args.title,
    legend=args.legend,
  )


if __name__ == "__main__":
  main()
