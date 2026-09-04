#!/usr/bin/env python3
"""Pick the part of a reach the figure should show -- small matplotlib GUI.

A captured scene holds up to half a minute of tip path: the arm settling, the
reach itself, the hold afterwards, and whatever happened before you pressed the
button. Only a slice of that is the figure, and *which* slice is a decision
nobody can make while the arm is still moving -- so it is made here, afterwards,
by dragging.

Two panels, both live:

* **left** the figure as it will be drawn -- workspace, arm, target, tip -- with
  the discarded part of the path in grey and the kept part in purple, so you are
  choosing on the picture you will get and not on a proxy;
* **right** the same path over time (distance to the target, and x/z), which is
  where a reach is actually readable: the drop is the reach, the flat tail is
  the hold. Drag anywhere in it to select.

The choice is stored as ``<scene>_crop.json`` next to the scene, and
:mod:`spirob_rl.rig.target_figure` picks it up automatically -- so a crop is set once
and every later render of that scene uses it (``--trace-start``/``--trace-end``
still win, ``--ignore-crop-file`` disables it). Saving also re-renders the
figure straight away, in both variants (with and without the path).

Usage::

    uv run python -m spirob_rl.rig.target_crop --scene build/rl/figures/reach_*.json
    uv run python -m spirob_rl.rig.target_crop --listen --duration 20   # aufnehmen und gleich zuschneiden
    uv run python -m spirob_rl.rig.target_crop --scene ... --backend webagg   # remote

Controls: drag to select, ``f`` full, ``r`` the automatic reach, ``s`` save,
``q`` close.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Optional

from . import target_figure
from .target_figure import Scene

_KEPT = target_figure._TRACE
_DROPPED = "#b9b9b4"
_SELECTION = "#f5a623"


def reach_window(scene: Scene, settle_fraction: float = 0.05) -> tuple[float, float]:
  """The automatic proposal: from the last time the tip was far away to the end.

  "Far away" is measured against the reach's own size, so it works for a 5 cm
  correction and a 30 cm swing alike: walk back from the end while the tip is
  still closing in, and start where it last was more than 95 % of the way out.
  A reach that never converged yields the whole path, which is the honest
  answer -- there is no reach in it to find.
  """
  times = scene.trace_times
  if len(scene.trace) < 2:
    return (0.0, 0.0)
  distances = [math.hypot(x - scene.target[0], z - scene.target[1]) for x, z in scene.trace]
  far, near = max(distances), min(distances)
  if far - near < 1e-4:
    return (times[0], times[-1])
  threshold = near + (far - near) * (1.0 - settle_fraction)
  start = 0
  for index, distance in enumerate(distances):
    if distance >= threshold:
      start = index
  return (times[start], times[-1])


def run_gui(scene: Scene, scene_path: Optional[Path], initial: tuple[float, float]) -> None:
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Button, SpanSelector

  times = scene.trace_times
  selection = list(initial)
  distances = [math.hypot(x - scene.target[0], z - scene.target[1]) for x, z in scene.trace]

  figure = plt.figure(figsize=(14, 7))
  figure.canvas.manager.set_window_title(
    f"Bahn zuschneiden -- {scene_path.name if scene_path else 'Aufnahme'}"
  )
  ax_plane = figure.add_axes([0.04, 0.16, 0.44, 0.78])
  ax_dist = figure.add_axes([0.56, 0.56, 0.41, 0.38])
  ax_xz = figure.add_axes([0.56, 0.16, 0.41, 0.34], sharex=ax_dist)

  # -- right: the path over time, which is what gets dragged on ----------------
  ax_dist.plot(times, [d * 1000 for d in distances], color=_KEPT, lw=1.4)
  ax_dist.set_ylabel("Abstand zum Ziel [mm]")
  ax_dist.grid(True, ls=":", alpha=0.5)
  ax_xz.plot(times, [p[0] for p in scene.trace], color="#0071b2", lw=1.2, label="x")
  ax_xz.plot(times, [p[1] for p in scene.trace], color="#d55c00", lw=1.2, label="z")
  ax_xz.set_ylabel("Spitze [m]")
  ax_xz.set_xlabel("Zeit seit Aufnahmebeginn [s]")
  ax_xz.grid(True, ls=":", alpha=0.5)
  ax_xz.legend(loc="best", fontsize=8)

  highlight = [
    ax.axvspan(selection[0], selection[1], alpha=0.20, color=_SELECTION, zorder=0)
    for ax in (ax_dist, ax_xz)
  ]

  def move_span(patch, lo: float, hi: float) -> None:
    """Move an axvspan. Matplotlib >= 3.9 returns a Rectangle here, older
    versions a Polygon -- the two want different setters."""
    if hasattr(patch, "set_width"):
      patch.set_x(lo)
      patch.set_width(hi - lo)
    else:
      patch.set_xy([[lo, 0.0], [lo, 1.0], [hi, 1.0], [hi, 0.0], [lo, 0.0]])

  # -- left: the figure itself, redrawn on every change ------------------------
  def draw_plane() -> None:
    """Repaint the preview: the whole path in grey, the kept part on top.

    Deliberately the real drawing code from ``target_figure`` rather than a
    lookalike -- what you crop against is then the figure you will get, and the
    two cannot drift apart.
    """
    ax_plane.clear()
    kept = scene.cropped(*selection)
    target_figure.draw_workspace(ax_plane, language=target_figure._default_language())
    ax_plane.plot(  # the whole path in grey, the kept part drawn over it
      [p[0] for p in scene.trace], [p[1] for p in scene.trace],
      color=_DROPPED, lw=1.6, zorder=2.6, solid_capstyle="round",
    )
    target_figure.draw_measurement(
      ax_plane, kept, language=target_figure._default_language(), annotate=False
    )
    # The full scene frames the view, so it does not jump while dragging.
    target_figure.autoscale(ax_plane, scene)
    ax_plane.grid(True, alpha=0.4)
    ax_plane.set_axisbelow(True)
    ax_plane.set_xlabel("x [m]")
    ax_plane.set_ylabel("z [m]")
    ax_plane.set_title(
      f"Bahn {selection[0]:.2f} - {selection[1]:.2f} s   "
      f"({selection[1] - selection[0]:.2f} s, {len(kept.trace)} Punkte)",
      fontsize=10,
    )

  def refresh() -> None:
    for patch in highlight:
      move_span(patch, selection[0], selection[1])
    draw_plane()
    figure.canvas.draw_idle()

  def on_select(lo: float, hi: float) -> None:
    if hi - lo < 1e-3:
      return
    selection[:] = [float(lo), float(hi)]
    refresh()

  selector = SpanSelector(
    ax_dist, on_select, "horizontal", useblit=False, interactive=True, drag_from_anywhere=True,
    props={"facecolor": _SELECTION, "alpha": 0.25},
  )
  selector.extents = tuple(selection)

  def use(window: tuple[float, float]) -> None:
    selection[:] = [float(window[0]), float(window[1])]
    selector.extents = tuple(selection)
    refresh()

  def save(_event=None) -> None:
    kept = scene.cropped(*selection)
    if scene_path is not None:
      path = target_figure.save_crop_file(scene_path, selection[0], selection[1])
      print(f"[crop] {path}  ->  t_start={selection[0]:.3f}s t_end={selection[1]:.3f}s")
      print(f"[crop] entspricht: --trace-start {selection[0]:.3f} --trace-end {selection[1]:.3f}")
      base = scene_path.with_suffix("")
    else:
      base = target_figure.default_out_base("reach")
      kept.save(base.with_suffix(".json"))
    written = target_figure.save_figure(kept, out_base=base, write_scene=False)
    ax_plane.set_title(f"gespeichert: {written[0].name} (+ ohne Bahn)", fontsize=10)
    figure.canvas.draw_idle()

  buttons = []
  for offset, (label, callback) in enumerate((
    ("Reach", lambda _e: use(reach_window(scene))),
    ("Alles", lambda _e: use((times[0], times[-1]))),
    ("Speichern", save),
  )):
    axis = figure.add_axes([0.56 + offset * 0.10, 0.03, 0.09, 0.06])
    button = Button(axis, label)
    button.on_clicked(callback)
    buttons.append(button)  # keep references alive, else the callbacks die

  def on_key(event) -> None:
    if event.key == "f":
      use((times[0], times[-1]))
    elif event.key == "r":
      use(reach_window(scene))
    elif event.key == "s":
      save()

  figure.canvas.mpl_connect("key_press_event", on_key)
  refresh()
  print("[crop] ziehen zum Auswaehlen | r=Reach  f=alles  s=speichern | Fenster schliessen zum Beenden")
  plt.show()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--scene", type=Path, default=None,
                      help="Aufgenommene Szene (*.json), Glob erlaubt")
  parser.add_argument("--listen", action="store_true",
                      help="Erst aus der Bridge-Telemetrie aufnehmen, dann zuschneiden")
  parser.add_argument("--duration", type=float, default=20.0, help="Aufnahmefenster fuer --listen [s]")
  parser.add_argument("--telemetry-port", type=int, default=None)
  parser.add_argument("--backend", default=None,
                      help="matplotlib-Backend, z.B. tkagg oder webagg (webagg zeigt die GUI im Browser)")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.backend:
    import matplotlib

    matplotlib.use(args.backend)

  scene_path: Optional[Path] = None
  if args.listen:
    from . import telemetry

    scene = target_figure.capture_from_telemetry(
      args.duration, port=args.telemetry_port or telemetry.TELEMETRY_UDP_PORT
    )
    scene_path = target_figure.default_out_base("reach").with_suffix(".json")
    scene.save(scene_path)
    print(f"[crop] Aufnahme gespeichert: {scene_path}")
  elif args.scene is not None:
    matches = sorted(glob.glob(str(args.scene))) if any(c in str(args.scene) for c in "*?[") \
      else [args.scene]
    if not matches:
      raise SystemExit(f"Keine Szene passt auf {args.scene}")
    if len(matches) > 1:
      print(f"[crop] {len(matches)} Dateien passen, nehme {matches[-1]}")
    scene_path = Path(matches[-1])  # newest by name: captures are timestamped
    scene = Scene.load(scene_path)
  else:
    raise SystemExit("Entweder --scene <datei.json> oder --listen angeben")

  if len(scene.trace) < 2:
    raise SystemExit(
      "Diese Szene enthaelt keine Bahn -- es gibt nichts zuzuschneiden. "
      "(Lief die Bridge mit Winkelbrett, und hat sich der Arm bewegt?)"
    )

  times = scene.trace_times
  stored = target_figure.load_crop_file(scene_path) if scene_path else None
  initial = stored or reach_window(scene)
  print(
    f"[crop] {len(scene.trace)} Punkte ueber {times[-1] - times[0]:.2f} s, "
    f"Vorauswahl {initial[0]:.2f} - {initial[1]:.2f} s"
    + ("  (aus *_crop.json)" if stored else "  (automatisch erkannter Reach)")
  )

  try:
    run_gui(scene, scene_path, initial)
  except Exception as exc:
    raise SystemExit(
      f"GUI liess sich nicht oeffnen ({exc}).\n"
      "Versuche --backend webagg (Browser), oder schneide ohne GUI zu: "
      "target_figure --scene ... --trace-start X --trace-end Y"
    ) from exc


if __name__ == "__main__":
  main()
