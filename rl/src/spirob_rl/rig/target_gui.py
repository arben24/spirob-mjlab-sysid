"""Graphical TCP-target picker and position display for the SpiRob bridge.

Draws the reachable workspace shell in the robot's x-z plane, lets you move the
target with two sliders -- and shows where the arm actually ended up. Run it as
its own process, next to a running ``python -m spirob_rl.rig.policy_bridge``:

    uv run python -m spirob_rl.rig.target_gui

Two directions of traffic, both plain UDP on localhost:

* **out**: the target you pick, as ``"x z"`` text (port 47800).
* **in**: the bridge's per-tick telemetry (port 47801), which carries the
  measured tip position -- forward kinematics over the accelerometer board's
  joint angles. That is the only thing on the rig that knows where the arm is,
  so without that board connected the GUI honestly shows nothing rather than a
  stale dot.

Why a separate process: the control loop has a 20 ms budget per tick, and a GUI
in the same process would compete for it -- Tk redraws, slider drags and window
resizes all take unbounded time, and Python's GIL means even a background thread
would steal it from inference. Here the GUI can stall, be resized, or be killed
outright without the loop noticing.

Targets travel as plain ASCII ``"x z"`` UDP datagrams to localhost, the same
text format the bridge accepts on stdin. UDP is deliberate: sending never
blocks, the receiver applies no backpressure, and "latest datagram wins" is
exactly what a setpoint wants -- a dropped intermediate value during a drag is
irrelevant because another one follows milliseconds later.

The GUI is stateless with respect to the bridge: it re-sends the current target
about twice a second, so starting either process before the other, or
restarting the bridge mid-session, converges without any handshake.

For a document, do not photograph this window: "Abbildung speichern" (Ctrl+S)
hands the *raw numbers* on screen -- target, measured tip, arm pose, and the
recorded path of the tip -- to :mod:`spirob_rl.rig.target_figure`, which redraws them
as a vector figure (with and without the path) and stores them as JSON next to
it, so the same figure can be re-rendered later without going back to the rig;
:mod:`spirob_rl.rig.target_crop` then trims the path to the part worth showing.
matplotlib is imported only when that button is pressed; the GUI itself stays
stdlib-only and starts instantly.
"""

from __future__ import annotations

import argparse
import math
import socket
import tkinter as tk
from collections import deque
from datetime import datetime

from . import kinematics, telemetry, workspace

_VIEW_X = (-0.52, 0.52)  # metres of x visible on the canvas
_VIEW_Z = (-0.22, 0.56)  # metres of z visible on the canvas
_SHELL_SAMPLES = 121  # polygon resolution of the shell band
_SEND_INTERVAL_MS = 33  # ~30 Hz while dragging a slider
_HEARTBEAT_MS = 500  # re-send so a late-starting bridge picks the target up
_RECEIVE_INTERVAL_MS = 50  # 20 Hz is plenty for the eye; the bridge runs at 50
_STALE_AFTER_S = 1.0
"""Beyond this, the measured position is not shown at all.

A dot frozen where the arm was a minute ago is worse than no dot: it looks like
a measurement. Silence is the honest display for a bridge that stopped talking.
"""
_STATUS_COLS = 68  # fixed status width in characters, keeps the window stable
_TRACE_SECONDS = 30.0
"""How much of the tip's path the figure export can look back on.

Only kept for the figure -- the canvas draws the current pose, not the history.
At 50 Hz this is a few thousand points, and only ones the tip actually moved
between are stored.
"""
_TRACE_MIN_STEP_M = 0.002

_BG = "#12151a"
_GRID = "#2a313d"
_AXIS = "#5f6d80"
_SHELL_FILL = "#1d3a2e"
_SHELL_EDGE = "#48b184"
_ROBOT = "#6b7889"
_ARM = "#4aa3df"  # live arm pose: blue, distinct from shell (green) and tip (amber)
_OK = "#4ade80"
_BAD = "#f87171"
_ACTUAL = "#f0b429"  # measured tip: amber, distinct from both verdict colours

# Drawing weights, generous on purpose: this window is watched from across a
# desk and gets projected, and hairlines survive neither. It is still a *screen*
# budget -- anything that goes into a document should come out of
# ``target_figure`` via the export button, not out of a screenshot.
_W_GRID = 1
_W_AXIS = 2
_W_SHELL = 2
_W_CENTER = 3
_W_ARM = 8
_R_JOINT = 5
_R_BASE = 9
_W_TARGET = 3
_R_TARGET = 10
_L_CROSS = 22
_R_ACTUAL = 10
_FONT = 11
_FONT_SMALL = 10


class MeasuredPosition:
  """Holds the newest measured tip and arm shape, forgetting them when stale.

  Kept apart from the widgets so the rule that matters can be tested without a
  display: a position is only shown while it is *current*. A dot -- or an arm
  pose -- left over from a bridge that died a minute ago looks exactly like a
  live measurement, and that is the one failure mode a position display must
  not have.

  The arm shape is the segment points from forward kinematics over the measured
  joint angles (``signals["joint_pos"]``), so it and the tip share one freshness
  clock -- they come from the same telemetry packet.
  """

  def __init__(
    self, geometry: "kinematics.ChainGeometry | None" = None,
    stale_after_s: float = _STALE_AFTER_S,
  ) -> None:
    self.geometry = geometry
    self.stale_after_s = stale_after_s
    self.value: tuple[float, float] | None = None
    self.points: list[tuple[float, float]] | None = None
    self.age_s: float = float("inf")
    self.ever_received = False

  def update(self, states: list[telemetry.BridgeState]) -> bool:
    """Take the newest state. Returns True if the display needs a redraw."""
    if not states:
      return False
    state = states[-1]
    self.value = state.tcp_actual
    self.points = self._arm_from(state)
    self.age_s = 0.0
    self.ever_received = True
    return True

  def _arm_from(self, state: telemetry.BridgeState) -> list[tuple[float, float]] | None:
    """Segment points (base->tip) from the telemetry's joint angles, or None."""
    if self.geometry is None:
      return None
    q = state.signals.get("joint_pos")
    if q is None or len(q) != self.geometry.num_joints:
      return None
    pts = kinematics.forward_kinematics(q, self.geometry)
    return [(float(x), float(z)) for x, z in pts]

  def advance(self, elapsed_s: float) -> bool:
    """Let time pass. Returns True when a fresh value has just expired."""
    was_fresh = self.fresh is not None
    self.age_s += elapsed_s
    return was_fresh and self.fresh is None

  @property
  def fresh(self) -> tuple[float, float] | None:
    if self.value is None or self.age_s > self.stale_after_s:
      return None
    return self.value

  @property
  def arm(self) -> list[tuple[float, float]] | None:
    """The measured arm shape while it is current, else None."""
    if self.points is None or self.age_s > self.stale_after_s:
      return None
    return self.points


class TipTrace:
  """The recent path of the measured tip, kept for the figure export.

  Not drawn on the canvas -- that shows the present pose -- but a figure of a
  reach wants the way there. Two rules, both about not lying with a line:

  * a point is only stored once the tip actually *moved*, so a held pose becomes
    one point instead of a thousand stacked ones that draw as a blob;
  * anything older than ``seconds`` is dropped, so the exported path is the
    recent reach and not everything since the GUI was started.

  Timestamps come from the bridge's ``time.monotonic`` (same machine), so the
  window is real time and not "however often the GUI happened to redraw".
  Widget-free for the same reason as :class:`MeasuredPosition`: it can be tested
  without a display.
  """

  def __init__(self, seconds: float = _TRACE_SECONDS, min_step_m: float = _TRACE_MIN_STEP_M) -> None:
    self.seconds = seconds
    self.min_step_m = min_step_m
    self._samples: deque[tuple[float, tuple[float, float]]] = deque()

  def record(self, t: float, point: tuple[float, float]) -> None:
    if self._samples:
      last = self._samples[-1][1]
      if math.hypot(point[0] - last[0], point[1] - last[1]) < self.min_step_m:
        return
    self._samples.append((t, point))
    while self._samples and t - self._samples[0][0] > self.seconds:
      self._samples.popleft()

  @property
  def points(self) -> tuple[tuple[float, float], ...]:
    return tuple(point for _t, point in self._samples)

  @property
  def times(self) -> tuple[float, ...]:
    """Sample times relative to the first one -- what the crop GUI cuts on."""
    if not self._samples:
      return ()
    first = self._samples[0][0]
    return tuple(t - first for t, _point in self._samples)

  @property
  def span_s(self) -> float:
    """Wall-clock time the stored path covers; 0 for fewer than two points."""
    if len(self._samples) < 2:
      return 0.0
    return self._samples[-1][0] - self._samples[0][0]


class TargetGui:
  def __init__(
    self, host: str, port: int, x: float, z: float, telemetry_port: int = 0,
    monitor: bool = False, trace_seconds: float = _TRACE_SECONDS,
  ) -> None:
    # Monitor mode: the target is not chosen here but sampled by the training
    # env, which sends it in the telemetry. The GUI then only *shows* soll (that
    # target) and ist (the measured tip) -- it never sends. Sliders and buttons
    # are inert so it is obvious the arm is not being commanded from here.
    self.monitor = monitor
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.address = (host, port)
    self.last_sent: tuple[float, float] | None = None
    self.send_pending = not monitor
    self._sliders: list[tk.Scale] = []

    # Where the arm actually is, as reported by the bridge. "Nobody told us" is
    # a different thing from "at the origin" and is drawn differently: not at
    # all. The chain geometry lets it also draw the whole arm from the measured
    # joint angles; if the model cannot be read, the tip dot still works.
    try:
      geometry = kinematics.load_chain_geometry()
    except Exception as exc:
      geometry = None
      print(f"[WARN] could not load chain geometry, arm will not be drawn: {exc}")
    self.measured = MeasuredPosition(geometry=geometry)
    # For the figure export only, never for the canvas. Recorded generously: the
    # window that ends up in the figure is chosen afterwards in target_crop, and
    # only what was recorded can be chosen from.
    self.trace = TipTrace(seconds=trace_seconds)
    self.receiver: telemetry.TelemetryReceiver | None = None
    self.receive_error: str | None = None
    if telemetry_port:
      try:
        self.receiver = telemetry.TelemetryReceiver(port=telemetry_port)
      except OSError as exc:
        # Not fatal: picking targets works without ever hearing back.
        self.receive_error = str(exc)

    self.root = tk.Tk()
    self.root.title("SpiRob TCP target")
    self.root.configure(bg=_BG)
    self.root.minsize(700, 660)  # below this the button row starts losing buttons

    self.canvas = tk.Canvas(self.root, bg=_BG, highlightthickness=0, height=430)
    self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
    self.canvas.bind("<Configure>", lambda _e: self.redraw())
    # Clicking the plane is often quicker than two sliders; the sliders stay
    # authoritative and are updated from here.
    self.canvas.bind("<Button-1>", self._on_canvas_click)
    self.canvas.bind("<B1-Motion>", self._on_canvas_click)

    self.x_var = tk.DoubleVar(value=x)
    self.z_var = tk.DoubleVar(value=z)
    self._add_slider("x  (seitwärts)", self.x_var, _VIEW_X)
    self._add_slider("z  (nach oben)", self.z_var, _VIEW_Z)

    button_row = tk.Frame(self.root, bg=_BG)
    button_row.pack(fill=tk.X, padx=12, pady=(2, 4))
    btn_state = tk.DISABLED if monitor else tk.NORMAL
    tk.Button(
      button_row, text="Auf Schale einrasten", command=self._snap, bg="#222833", fg="#dde3ec",
      activebackground="#2c3444", activeforeground="#ffffff", relief=tk.FLAT, padx=10,
      state=btn_state,
    ).pack(side=tk.LEFT)
    tk.Button(
      button_row, text="Zurück auf (0, 0.43)", command=self._reset, bg="#222833", fg="#dde3ec",
      activebackground="#2c3444", activeforeground="#ffffff", relief=tk.FLAT, padx=10,
      state=btn_state,
    ).pack(side=tk.LEFT, padx=(8, 0))
    # Always enabled, monitor mode included -- capturing is not commanding, and
    # watching a training run is exactly when a figure is worth having.
    tk.Button(
      button_row, text="Abbildung speichern (Ctrl+S)", command=self._save_figure,
      bg="#243043", fg="#cfe2f7", activebackground="#2f3f58", activeforeground="#ffffff",
      relief=tk.FLAT, padx=10,
    ).pack(side=tk.RIGHT)
    self.root.bind("<Control-s>", lambda _e: self._save_figure())

    # Fixed character width and height: a Label sizes its toplevel to its text,
    # so a status string that grows when the target leaves the shell would make
    # the whole window jump around mid-drag.
    self.status = tk.Label(
      self.root, text="", bg=_BG, fg="#dde3ec", font=("TkFixedFont", 10),
      justify=tk.LEFT, anchor="w", width=_STATUS_COLS, height=3,
    )
    self.status.pack(fill=tk.X, padx=12, pady=(0, 4))

    start_text = (
      "Monitor-Modus: Ziel kommt vom Training (nur Anzeige)"
      if monitor
      else f"→ udp://{host}:{port}"
    )
    self.sent_label = tk.Label(
      self.root, text=start_text, bg=_BG, fg="#6b7688",
      font=("TkFixedFont", 9), anchor="w",
    )
    self.sent_label.pack(fill=tk.X, padx=12, pady=(0, 2))

    # Its own line: the send label is rewritten twice a second by the heartbeat,
    # so a path written there would be gone before it could be read.
    self.figure_label = tk.Label(
      self.root, text="", bg=_BG, fg="#6b7688", font=("TkFixedFont", 9), anchor="w",
    )
    self.figure_label.pack(fill=tk.X, padx=12, pady=(0, 10))

    # In monitor mode nothing is ever sent, so the send pumps stay off.
    if not monitor:
      self.root.after(_SEND_INTERVAL_MS, self._pump_send)
      self.root.after(_HEARTBEAT_MS, self._heartbeat)
    self.root.after(_RECEIVE_INTERVAL_MS, self._pump_receive)
    self.redraw()

  # -- widgets ---------------------------------------------------------------

  def _add_slider(self, label: str, var: tk.DoubleVar, value_range: tuple[float, float]) -> None:
    frame = tk.Frame(self.root, bg=_BG)
    frame.pack(fill=tk.X, padx=12)
    tk.Label(
      frame, text=label, bg=_BG, fg="#9aa5b5", width=16, anchor="w",
      font=("TkFixedFont", 10),
    ).pack(side=tk.LEFT)
    scale = tk.Scale(
      frame, variable=var, from_=value_range[0], to=value_range[1], resolution=0.005,
      orient=tk.HORIZONTAL, command=lambda _v: self._on_change(), bg=_BG, fg="#dde3ec",
      troughcolor="#1b2029", highlightthickness=0, activebackground=_OK, borderwidth=0,
      state=tk.DISABLED if self.monitor else tk.NORMAL,
    )
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self._sliders.append(scale)

  # -- interaction -----------------------------------------------------------

  def _on_change(self) -> None:
    self.send_pending = True
    self.redraw()

  def _on_canvas_click(self, event) -> None:
    x, z = self._to_world(event.x, event.y)
    self.x_var.set(round(max(_VIEW_X[0], min(_VIEW_X[1], x)), 3))
    self.z_var.set(round(max(_VIEW_Z[0], min(_VIEW_Z[1], z)), 3))
    self._on_change()

  def _snap(self) -> None:
    x, z = workspace.snap_to_shell(self.x_var.get(), self.z_var.get())
    self.x_var.set(round(x, 3))
    self.z_var.set(round(z, 3))
    self._on_change()

  def _reset(self) -> None:
    self.x_var.set(0.0)
    self.z_var.set(workspace.SHELL_RADIUS)
    self._on_change()

  # -- sending ---------------------------------------------------------------

  def _pump_send(self) -> None:
    if self.send_pending:
      target = (self.x_var.get(), self.z_var.get())
      self._send(target)
      self.send_pending = False
    self.root.after(_SEND_INTERVAL_MS, self._pump_send)

  def _heartbeat(self) -> None:
    self._send((self.x_var.get(), self.z_var.get()))
    self.root.after(_HEARTBEAT_MS, self._heartbeat)

  # -- receiving -------------------------------------------------------------

  def _pump_receive(self) -> None:
    """Take the newest bridge tick and redraw only if something changed.

    Draining is non-blocking and only the last datagram matters -- the bridge
    sends 50 per second and the display needs the current one, not a backlog.
    """
    if self.receiver is not None:
      states = self.receiver.drain()
      for state in states:
        if state.tcp_actual is not None:
          self.trace.record(state.t, state.tcp_actual)
      received = self.measured.update(states)
      # Monitor mode: follow the target the training env sampled, so the
      # existing crosshair drawing shows soll while the dot shows ist.
      if self.monitor and states:
        tx, tz = states[-1].target
        self.x_var.set(round(tx, 4))
        self.z_var.set(round(tz, 4))
        received = True
      expired = self.measured.advance(_RECEIVE_INTERVAL_MS / 1000.0)
      if received or expired:
        self.redraw()  # on new data, and once more when the dot goes away
    self.root.after(_RECEIVE_INTERVAL_MS, self._pump_receive)

  # -- figure export ---------------------------------------------------------

  def _save_figure(self) -> None:
    """Hand the current raw numbers to ``target_figure`` and report where they went.

    matplotlib is imported here rather than at module level so the GUI keeps its
    instant start, and every failure is caught: an export that raises inside a Tk
    callback would otherwise take the window down mid-experiment.
    """
    self.figure_label.config(text="Abbildung wird gezeichnet ...", fg="#6b7688")
    self.figure_label.update_idletasks()
    try:
      from . import target_figure
    except Exception as exc:  # matplotlib missing, most likely
      self.figure_label.config(text=f"! Abbildung nicht moeglich: {exc}", fg=_BAD)
      return

    scene = target_figure.Scene(
      target=(self.x_var.get(), self.z_var.get()),
      tcp_actual=self.measured.fresh,
      arm=tuple(self.measured.arm or ()),
      trace=self.trace.points,
      trace_t=self.trace.times,
      trace_seconds=self.trace.span_s,
      from_rig=self.receiver is not None,
      captured_at=datetime.now().isoformat(timespec="seconds"),
      note="Monitor-Modus" if self.monitor else "",
    )
    try:
      written = target_figure.save_figure(scene)
    except Exception as exc:
      self.figure_label.config(text=f"! Abbildung fehlgeschlagen: {exc}", fg=_BAD)
      return
    base = written[0].with_suffix("")
    variants = "mit + ohne Bahn" if len(scene.trace) > 1 else "ohne Bahn (keine Telemetrie)"
    self.figure_label.config(text=f"{base}.*  ({variants}, Zuschnitt: target_crop)", fg=_OK)

  def _send(self, target: tuple[float, float]) -> None:
    try:
      self.sock.sendto(f"{target[0]:.4f} {target[1]:.4f}".encode("ascii"), self.address)
      self.last_sent = target
      self.sent_label.config(
        text=f"→ udp://{self.address[0]}:{self.address[1]}   "
        f"gesendet: {target[0]:+.3f} {target[1]:+.3f}",
        fg="#6b7688",
      )
    except OSError as exc:
      self.sent_label.config(text=f"! Senden fehlgeschlagen: {exc}", fg=_BAD)

  # -- coordinate mapping ----------------------------------------------------

  def _transform(self) -> tuple[float, float, float]:
    """Return (scale px/m, x offset, y offset) preserving aspect ratio."""
    width = max(self.canvas.winfo_width(), 1)
    height = max(self.canvas.winfo_height(), 1)
    span_x = _VIEW_X[1] - _VIEW_X[0]
    span_z = _VIEW_Z[1] - _VIEW_Z[0]
    scale = min(width / span_x, height / span_z)
    off_x = (width - span_x * scale) / 2.0 - _VIEW_X[0] * scale
    off_y = (height - span_z * scale) / 2.0 + _VIEW_Z[1] * scale
    return scale, off_x, off_y

  def _to_px(self, x: float, z: float) -> tuple[float, float]:
    scale, off_x, off_y = self._transform()
    return x * scale + off_x, off_y - z * scale

  def _to_world(self, px: float, py: float) -> tuple[float, float]:
    scale, off_x, off_y = self._transform()
    return (px - off_x) / scale, (off_y - py) / scale

  # -- drawing ---------------------------------------------------------------

  def redraw(self) -> None:
    c = self.canvas
    c.delete("all")
    self._draw_grid()
    self._draw_shell()
    self._draw_arm()
    self._draw_target()
    self._draw_actual()  # after the target, so the deviation line sits on top
    self._update_status()

  def _draw_grid(self) -> None:
    c = self.canvas
    step = 0.1
    ticks = [i * step for i in range(-5, 6)]
    for value in ticks:
      if _VIEW_X[0] <= value <= _VIEW_X[1]:
        x0, y0 = self._to_px(value, _VIEW_Z[0])
        x1, y1 = self._to_px(value, _VIEW_Z[1])
        c.create_line(x0, y0, x1, y1, fill=_GRID, width=_W_GRID)
      if _VIEW_Z[0] <= value <= _VIEW_Z[1]:
        x0, y0 = self._to_px(_VIEW_X[0], value)
        x1, y1 = self._to_px(_VIEW_X[1], value)
        c.create_line(x0, y0, x1, y1, fill=_GRID, width=_W_GRID)

    ox, oy = self._to_px(0.0, 0.0)
    x1, _ = self._to_px(_VIEW_X[1], 0.0)
    _, y1 = self._to_px(0.0, _VIEW_Z[1])
    c.create_line(self._to_px(_VIEW_X[0], 0.0)[0], oy, x1, oy, fill=_AXIS, width=_W_AXIS)
    c.create_line(ox, self._to_px(0.0, _VIEW_Z[0])[1], ox, y1, fill=_AXIS, width=_W_AXIS)
    c.create_text(x1 - 12, oy - 12, text="x [m]", fill=_AXIS, font=("TkFixedFont", _FONT_SMALL))
    c.create_text(ox + 26, y1 + 12, text="z [m]", fill=_AXIS, font=("TkFixedFont", _FONT_SMALL))
    for value in ticks:
      if abs(value) < 1e-9:
        continue
      if _VIEW_X[0] <= value <= _VIEW_X[1]:
        px, _ = self._to_px(value, 0.0)
        c.create_text(px, oy + 13, text=f"{value:.1f}", fill=_AXIS,
                      font=("TkFixedFont", _FONT_SMALL))
      if _VIEW_Z[0] <= value <= _VIEW_Z[1]:
        _, py = self._to_px(0.0, value)
        c.create_text(ox - 22, py, text=f"{value:.1f}", fill=_AXIS,
                      font=("TkFixedFont", _FONT_SMALL))

  def _draw_shell(self) -> None:
    """The band of holdable TCP positions, plus its centerline and end caps."""
    c = self.canvas
    angles = [
      -workspace.ANGLE_LIMIT + i * (2 * workspace.ANGLE_LIMIT / (_SHELL_SAMPLES - 1))
      for i in range(_SHELL_SAMPLES)
    ]
    outer, inner, center = [], [], []
    for angle in angles:
      s, cs = math.sin(angle), math.cos(angle)
      r = workspace.shell_radius(angle)
      outer.extend(self._to_px((r + workspace.SHELL_BAND) * s, (r + workspace.SHELL_BAND) * cs))
      center.extend(self._to_px(r * s, r * cs))
      inner.append(self._to_px((r - workspace.SHELL_BAND) * s, (r - workspace.SHELL_BAND) * cs))

    polygon = outer + [coord for point in reversed(inner) for coord in point]
    c.create_polygon(polygon, fill=_SHELL_FILL, outline=_SHELL_EDGE, width=_W_SHELL)
    c.create_line(center, fill=_SHELL_EDGE, width=_W_CENTER, dash=(6, 4))

    for angle in (-workspace.ANGLE_LIMIT, workspace.ANGLE_LIMIT):
      x, z = workspace.shell_point(angle)
      c.create_line(*self._to_px(0.0, 0.0), *self._to_px(x, z), fill=_AXIS, dash=(3, 5))
      # Pushed well past the band end so the label clears the x-axis ticks.
      lx, lz = self._to_px(x * 1.20, z * 1.20 - 0.025)
      c.create_text(lx, lz, text=f"{angle:+.1f} rad", fill=_SHELL_EDGE,
                    font=("TkFixedFont", _FONT_SMALL))

    # Legend in the corner rather than a label on the arc: the arc's apex is
    # exactly where the target sits at rest, so a label there collides with it.
    c.create_rectangle(14, 10, 32, 26, fill=_SHELL_FILL, outline=_SHELL_EDGE, width=_W_SHELL)
    c.create_text(
      40, 18, anchor="w", fill=_SHELL_EDGE, font=("TkFixedFont", _FONT),
      text=f"erreichbare Schale   r = {workspace.SHELL_RADIUS} - "
      f"{workspace.SHELL_CURVATURE}*w^2  (+-{workspace.SHELL_BAND} m)",
    )

  def _draw_arm(self) -> None:
    """The arm: its live segment pose if measured, else a faint rest hint.

    A simplified stick figure -- one line through the segment origins base->tip
    plus a dot at each joint -- so the actual pose in space is readable at a
    glance. Falls back to the straight rest arm when no telemetry is current, so
    a dead bridge never leaves a frozen pose masquerading as live.
    """
    c = self.canvas
    base_x, base_y = self._to_px(0.0, 0.0)
    arm = self.measured.arm

    if arm is not None:
      pixels = [self._to_px(x, z) for x, z in arm]
      flat = [coord for point in pixels for coord in point]
      c.create_line(*flat, fill=_ARM, width=_W_ARM, capstyle=tk.ROUND, joinstyle=tk.ROUND)
      # A dot at every joint (skip the base and the tip: base has its own marker,
      # the tip carries the amber measured-position dot).
      for px, py in pixels[1:-1]:
        c.create_oval(px - _R_JOINT, py - _R_JOINT, px + _R_JOINT, py + _R_JOINT,
                      fill=_ARM, outline=_BG)
    else:
      tip_x, tip_y = self._to_px(0.0, workspace.SHELL_RADIUS)
      c.create_line(base_x, base_y, tip_x, tip_y, fill=_ROBOT, width=_W_ARM, capstyle=tk.ROUND)

    c.create_oval(base_x - _R_BASE, base_y - _R_BASE, base_x + _R_BASE, base_y + _R_BASE,
                  fill=_ROBOT, outline="")
    c.create_text(base_x - 38, base_y - 14, text="Basis", fill=_ROBOT,
                  font=("TkFixedFont", _FONT))

  def _draw_target(self) -> None:
    c = self.canvas
    x, z = self.x_var.get(), self.z_var.get()
    info = workspace.describe(x, z)
    color = _OK if info.ok else _BAD
    px, py = self._to_px(x, z)
    c.create_line(px - _L_CROSS, py, px + _L_CROSS, py, fill=color, width=_W_TARGET)
    c.create_line(px, py - _L_CROSS, px, py + _L_CROSS, fill=color, width=_W_TARGET)
    c.create_oval(px - _R_TARGET, py - _R_TARGET, px + _R_TARGET, py + _R_TARGET,
                  outline=color, width=_W_TARGET)
    # Flip the readout to the inside when the target sits near the right edge,
    # otherwise the text runs off the canvas at large +x.
    flip = px > self.canvas.winfo_width() - 130
    c.create_text(
      px - 16 if flip else px + 16, py - 20, text=f"({x:+.3f}, {z:+.3f})", fill=color,
      font=("TkFixedFont", _FONT), anchor="e" if flip else "w",
    )
    if not info.ok:
      sx, sz = workspace.snap_to_shell(x, z)
      c.create_line(px, py, *self._to_px(sx, sz), fill=_BAD, width=_W_TARGET, dash=(4, 3))

  def _draw_actual(self) -> None:
    """The measured tip, and the gap between it and the target.

    Drawn as a filled dot to set it apart from the target's open crosshair:
    the target is a wish, this is a measurement.
    """
    actual = self.measured.fresh
    if actual is None:
      return
    ax, ay = self._to_px(*actual)
    tx, ty = self._to_px(self.x_var.get(), self.z_var.get())
    c = self.canvas
    c.create_line(tx, ty, ax, ay, fill=_ACTUAL, width=_W_TARGET, dash=(5, 4))
    c.create_oval(ax - _R_ACTUAL, ay - _R_ACTUAL, ax + _R_ACTUAL, ay + _R_ACTUAL,
                  fill=_ACTUAL, outline=_BG, width=2)
    flip = ax > self.canvas.winfo_width() - 130
    c.create_text(
      ax - 18 if flip else ax + 18, ay + 18,
      text=f"ist ({actual[0]:+.3f}, {actual[1]:+.3f})", fill=_ACTUAL,
      font=("TkFixedFont", _FONT), anchor="e" if flip else "w",
    )

  def _error_text(self) -> tuple[str, str]:
    """Third status line: the deviation, or why there is none. (text, colour)"""
    if self.receiver is None:
      return ("ist-Position aus: mit --telemetry-port einschalten", "#6b7688")
    if self.receive_error is not None:
      return (f"ist-Position nicht empfangbar: {self.receive_error}", _BAD)
    actual = self.measured.fresh
    if actual is None:
      if not self.measured.ever_received:
        return ("ist-Position: keine Telemetrie - laeuft die Bridge?", "#6b7688")
      if self.measured.value is None:
        return ("ist-Position: kein Winkelsensor an der Bridge (--joint-port)", "#6b7688")
      return ("ist-Position: Telemetrie abgerissen", _BAD)
    error = math.hypot(actual[0] - self.x_var.get(), actual[1] - self.z_var.get())
    return (
      f"ist {actual[0]:+.3f} {actual[1]:+.3f}   abweichung {error * 1000:5.1f} mm",
      _ACTUAL,
    )

  def _update_status(self) -> None:
    """Three fixed-length lines: a verdict, the numbers, then soll gegen ist."""
    x, z = self.x_var.get(), self.z_var.get()
    info = workspace.describe(x, z)
    if info.ok:
      headline = "AUF DER SCHALE - Ziel liegt im trainierten Bereich"
      color = _OK
    elif not info.angle_ok:
      headline = f"AUSSERHALB - Winkel jenseits +-{workspace.ANGLE_LIMIT:.1f} rad"
      color = _BAD
    else:
      headline = "AUSSERHALB - Radius neben der Schale, nicht haltbar"
      color = _BAD
    numbers = (
      f"winkel {info.angle:+.2f} rad   radius {info.radius:.3f} m   "
      f"soll {info.shell_radius:.3f}+-{workspace.SHELL_BAND:.2f} m   "
      f"abw {info.deviation:+.3f} m"
    )
    error_text, _error_colour = self._error_text()
    # One label, so the window cannot change height when the bridge appears or
    # goes away. The verdict colour stays the label's; the deviation line reads
    # as part of the same block.
    self.status.config(text=f"{headline}\n{numbers}\n{error_text}", fg=color)

  def run(self) -> None:
    self.root.mainloop()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--host", default="127.0.0.1", help="Where the policy bridge listens")
  parser.add_argument("--port", type=int, default=workspace.TARGET_UDP_PORT, help="UDP port")
  parser.add_argument("--target-x", type=float, default=0.0, help="Initial x [m]")
  parser.add_argument("--target-z", type=float, default=workspace.SHELL_RADIUS, help="Initial z [m]")
  parser.add_argument(
    "--telemetry-port", type=int, default=telemetry.TELEMETRY_UDP_PORT,
    help="UDP port the bridge publishes its per-tick state on. The measured tip position "
    "comes from there. 0 turns the display off.",
  )
  parser.add_argument(
    "--trace-seconds", type=float, default=_TRACE_SECONDS,
    help="Wie viel Bahn der Spitze fuer die Abbildung mitgeschrieben wird [s]. Der "
    "tatsaechlich gezeigte Ausschnitt wird spaeter in spirob_rl.rig.target_crop gewaehlt.",
  )
  parser.add_argument(
    "--monitor", action="store_true",
    help="Watch online training: show the target the training env sampled (soll) and the "
    "measured tip (ist). Read-only -- sliders/buttons are inert, nothing is sent.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  TargetGui(
    args.host, args.port, args.target_x, args.target_z,
    telemetry_port=args.telemetry_port, monitor=args.monitor,
    trace_seconds=args.trace_seconds,
  ).run()


if __name__ == "__main__":
  main()
