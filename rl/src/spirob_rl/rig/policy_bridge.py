"""Policy-to-hardware bridge for the SpiRob rig, with pluggable hardware.

Counterpart to ``inverted_pendulum/policy_bridge.py``, but built around three
decoupled layers so one bridge drives many hardware configurations:

* ``sources.py`` -- each hardware device (one USB port, one protocol) as a
  ``HardwareSource`` exposing named signals. The motor MCU is always present
  (it is the actuator); the joint-angle board is optional.
* ``observation.py`` -- the actor observation layout is read straight off the
  checkpoint's task, and each term is filled from the right source signal. What
  the policy needs decides which hardware must be connected; a mismatch fails
  loudly before any motion.
* this file -- the 50 Hz control loop, target input (CLI / stdin / GUI), the
  measured tip position, and timing supervision.

To try a different hardware configuration you connect different sources and
load the matching policy; the software validates that they fit. Example:

    # tendon policy, motor MCU only
    python -m spirob_rl.rig.policy_bridge RlExplor-Spirob-Tcp-Reach --port /dev/ttyUSB0

    # imu or joint-angle policy, motor MCU + accelerometer board on a second port
    python -m spirob_rl.rig.policy_bridge RlExplor-Spirob-Tcp-Reach-Imu \
        --port /dev/ttyUSB0 --joint-port /dev/ttyUSB1

Soll against ist: the rig has no sensor for its own tip, but the accelerometer
board plus the model's segment lengths give it away (``kinematics.py``). With
that board connected, every telemetry line carries the commanded target, the
measured position and the distance between them -- the quantity the task is
rewarded on, which the sim marks as privileged precisely because the hardware
cannot sense it directly. The same numbers go out over UDP for ``target_gui``.

Physical assumptions the sim side cannot verify -- confirm on the first run
(ideally with ``--dry-run`` first). Details on the relevant source in
``sources.py``: the tendon-length sign and zero reference (``MotorRig``,
``--tendon-sign`` / ``--null-tendons``) and the joint order and sign
(``JointSensor``, ``--joint-sign`` / ``--joint-order``).
"""

from __future__ import annotations

import argparse
import math
import select
import socket
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from spirob_rl.cli import LOG_ROOT
from spirob_rl.infer import InferConfig, load_policy

from . import kinematics, observation, telemetry, workspace
from .serial_protocol import find_default_port
from .sources import JointSensor, MotorRig

_MONITOR_WINDOW = 250  # rolling stats window (~5 s at 50 Hz)
_SUMMARY_INTERVAL_S = 5.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
    "task",
    nargs="?",
    default="RlExplor-Spirob-Tcp-Reach",
    help="RL task id (must be a 'force' or 'tendon' sensor-level SpiRob task)",
  )
  parser.add_argument("--checkpoint-file", help="Explicit checkpoint path")
  parser.add_argument(
    "--log-root",
    default=str(LOG_ROOT),
    help="Root directory for training runs (default: build/rl/logs)",
  )
  parser.add_argument("--device", help="Torch device, for example cpu or cuda:0")
  parser.add_argument("--port", help="Motor MCU serial port, for example /dev/ttyUSB0")
  parser.add_argument("--baudrate", type=int, default=460800, help="Motor MCU baud rate")
  parser.add_argument(
    "--joint-port",
    help="Joint-angle board serial port (second USB device). Required only for "
    "joint-angle policies; ignored otherwise.",
  )
  parser.add_argument("--joint-baud", type=int, default=500000, help="Joint-angle board baud rate")
  parser.add_argument(
    "--joint-sign",
    type=float,
    default=1.0,
    help="+1 or -1: flips the measured joint-angle sign to match the sim's bending convention.",
  )
  parser.add_argument(
    "--joint-order",
    default=None,
    help="Comma-separated 13-index permutation mapping sensor angles (base->tip) to the "
    "sim joint order. Default: identity (both are base->tip). See JointSensor.",
  )
  parser.add_argument("--poll-hz", type=float, default=50.0, help="Control-loop rate (matches training decimation)")
  parser.add_argument("--target-x", type=float, default=0.0, help="Initial TCP target x [m], base frame")
  parser.add_argument(
    "--target-z",
    type=float,
    default=workspace.SHELL_RADIUS,
    help="Initial TCP target z [m], base frame (default: straight up on the trained shell)",
  )
  parser.add_argument(
    "--target-port",
    type=int,
    default=workspace.TARGET_UDP_PORT,
    help="localhost UDP port to receive targets from spirob_rl.rig.target_gui on. 0 disables it.",
  )
  parser.add_argument(
    "--print-hz",
    type=float,
    default=10.0,
    help="Rate of the per-tick telemetry line. The control loop always runs at --poll-hz; "
    "this only throttles console output so it stays readable and typeable. 0 disables it "
    "(overruns and the periodic [LOOP] summary are always printed).",
  )
  parser.add_argument(
    "--telemetry-port",
    type=int,
    default=telemetry.TELEMETRY_UDP_PORT,
    help="localhost UDP port to publish per-tick state on. The target GUI reads the "
    "measured tip position from it. 0 disables it.",
  )
  parser.add_argument(
    "--telemetry-host",
    default="127.0.0.1",
    help="Host to publish telemetry to (the observer is meant to run on this machine).",
  )
  parser.add_argument(
    "--tendon-sign",
    default="1.0,1.0",
    help="Comma-separated +-1 per tendon to flip the rope-length sign convention (see module docstring)",
  )
  parser.add_argument(
    "--null-tendons",
    action="store_true",
    help="Send 'n all' right after connecting to zero rope length at the current pose. "
    "Only use this if the arm is physically straight (rest pose) right now.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Compute observations/actions and print them, but never send 'start'/'f' to the rig",
  )
  return parser.parse_args()


def _announce_target(x: float, z: float, previous_ok: bool, verbose: bool) -> bool:
  """Report a new TCP target and return whether it is inside the trained shell.

  The target is always accepted -- being off the shell only means the policy is
  extrapolating (on a tendon robot that usually looks like both tendons
  saturating), which is worth knowing but not worth refusing.

  Warnings fire only on the transition in or out of the trained workspace: a
  dragged GUI slider pushes ~30 targets a second, and re-printing the same
  warning for each would bury the telemetry.
  """
  messages = workspace.warnings_for(x, z)
  ok = not messages
  if verbose:
    print(f"[INFO] TCP target -> ({x:+.3f}, {z:+.3f}) m")
  if ok != previous_ok:
    if ok:
      print(f"[INFO] target ({x:+.3f}, {z:+.3f}) is back inside the trained shell")
    else:
      for message in messages:
        print(f"[WARN] {message}")
  return ok


def _format_tcp(target: list[float], actual: tuple[float, float] | None) -> str:
  """The line's central claim: where the tip should be, and where it is.

  Without the accelerometer board there is no measured position, and the field
  says so instead of leaving a plausible-looking gap.
  """
  soll = f"soll=({target[0]:+.3f},{target[1]:+.3f})"
  if actual is None:
    return f"{soll} ist=(  --  ,  --  ) err= --"
  error = math.hypot(actual[0] - target[0], actual[1] - target[1])
  return f"{soll} ist=({actual[0]:+.3f},{actual[1]:+.3f}) err={error * 1000:5.1f}mm"


def _drain_target_socket(sock: socket.socket) -> tuple[float, float] | None:
  """Return the newest target datagram, or None if none arrived.

  Non-blocking by construction: the socket never waits, and intermediate
  datagrams from a slider drag are intentionally discarded -- only the most
  recent setpoint matters.
  """
  latest: tuple[float, float] | None = None
  while True:
    try:
      data = sock.recv(64)
    except (BlockingIOError, InterruptedError):
      break
    except OSError:
      break
    parsed = workspace.parse_target(data.decode("ascii", errors="replace"))
    if parsed is not None:
      latest = parsed
  return latest


@dataclass
class LoopMonitor:
  """Measures whether the control loop actually holds its nominal rate.

  The policy was trained at a fixed 50 Hz (``decimation=5`` at a 0.004 s sim
  timestep), and it consumes velocities plus a 5-step observation history --
  so the *real* tick period is part of the sim-to-real gap, not just a
  cosmetic detail. Three numbers, easy to confuse:

  * ``period``: wall time between consecutive policy evaluations. This is
    what the policy actually experiences and what should equal ``1/poll_hz``.
  * ``work``: time spent inside one tick (serial drain + inference + tx).
    While ``work < period`` the surplus is idle sleep and the rate is
    holdable; once work exceeds the period the loop physically cannot keep up.
  * ``overruns``: scheduled slots missed outright. The scheduler stays on its
    original grid and skips missed slots rather than letting the phase drift,
    so this counts genuinely dropped control steps.
  """

  period_s: float
  periods_ms: deque[float] = field(
    default_factory=lambda: deque(maxlen=_MONITOR_WINDOW)
  )
  works_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_MONITOR_WINDOW))
  ticks: int = 0
  overruns: int = 0
  _last_begin: float | None = None
  _begin: float = 0.0

  def begin(self) -> float:
    now = time.monotonic()
    if self._last_begin is not None:
      self.periods_ms.append((now - self._last_begin) * 1000.0)
    self._last_begin = now
    self._begin = now
    self.ticks += 1
    return now

  def end(self) -> float:
    work_ms = (time.monotonic() - self._begin) * 1000.0
    self.works_ms.append(work_ms)
    return work_ms

  @property
  def last_period_ms(self) -> float:
    return self.periods_ms[-1] if self.periods_ms else float("nan")

  def summary(self) -> str:
    if not self.periods_ms:
      return "[LOOP] no timing data yet"
    nominal_ms = self.period_s * 1000.0
    mean_period = statistics.fmean(self.periods_ms)
    worst_jitter = max(abs(p - nominal_ms) for p in self.periods_ms)
    max_work = max(self.works_ms) if self.works_ms else float("nan")
    mean_work = statistics.fmean(self.works_ms) if self.works_ms else float("nan")
    return (
      f"[LOOP] {len(self.periods_ms)} ticks | "
      f"period {mean_period:.2f}ms avg "
      f"[{min(self.periods_ms):.2f}..{max(self.periods_ms):.2f}] "
      f"vs {nominal_ms:.2f} nominal -> {1000.0 / mean_period:.2f}Hz | "
      f"worst jitter {worst_jitter:.2f}ms | "
      f"work {mean_work:.2f}ms avg, {max_work:.2f}ms max "
      f"({100.0 * max_work / nominal_ms:.0f}% of budget) | "
      f"overruns {self.overruns}/{self.ticks}"
    )


def _parse_joint_order(text: str | None, num_joints: int) -> tuple[int, ...] | None:
  if text is None:
    return None
  try:
    order = tuple(int(v) for v in text.split(","))
  except ValueError:
    raise SystemExit(f"--joint-order must be comma-separated integers, got {text!r}")
  if sorted(order) != list(range(num_joints)):
    raise SystemExit(
      f"--joint-order must be a permutation of 0..{num_joints - 1}, got {text!r}"
    )
  return order


def _build_sources(args, layout: observation.ActorLayout) -> tuple[MotorRig, list]:
  """Construct (but do not open) the hardware sources this run needs.

  The motor rig is always built (it is the actuator). The joint sensor is built
  whenever a port is given -- even for a policy that does not read it, because
  it is also the rig's only ground truth: the forward kinematics over its
  angles is what turns "the policy was told to go there" into "the arm is
  here". A policy that *needs* it without a port being given is left for
  ``observation.validate`` to report with the canonical message.
  """
  motor_port = args.port or find_default_port()
  if motor_port is None:
    raise SystemExit("No motor serial port found. Pass --port /dev/ttyUSB0 (or similar).")
  tendon_sign = tuple(float(s) for s in args.tendon_sign.split(","))
  motor = MotorRig(
    port=motor_port,
    baudrate=args.baudrate,
    tendon_sign=tendon_sign,
    dry_run=args.dry_run,
    null_on_start=args.null_tendons,
  )
  sources: list = [motor]

  needs_joints = bool(observation.required_signals(layout) & set(JointSensor.provides))
  if args.joint_port:
    order = _parse_joint_order(args.joint_order, 13)
    sources.append(
      JointSensor(port=args.joint_port, baudrate=args.joint_baud, sign=args.joint_sign, order=order)
    )
    if not needs_joints:
      print(
        f"[INFO] Task {args.task!r} does not read the joint sensor, but it is connected: "
        "used as ground truth for the measured tip position only."
      )
  return motor, sources


def main() -> None:
  args = parse_args()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  policy, checkpoint_path = load_policy(
    args.task,
    InferConfig(checkpoint_file=args.checkpoint_file, log_root=args.log_root, device=args.device),
  )

  # The checkpoint's task is the source of truth for the observation layout.
  layout = observation.derive_actor_layout(args.task, device)
  policy_dim = getattr(policy, "obs_dim", None)
  if policy_dim is not None and policy_dim != layout.total_dim:
    raise SystemExit(
      f"Derived actor layout ({layout.total_dim} dim) disagrees with the checkpoint "
      f"({policy_dim} dim) -- layout derivation and the policy are out of sync."
    )

  motor, sources = _build_sources(args, layout)
  observation.validate(layout, sources)  # fails loudly if hardware can't satisfy the policy

  # The rig cannot sense its own tip -- but with the joint angles from the
  # accelerometer board and the segment lengths from the model, it follows.
  # That closes the loop the task is actually scored on: commanded target
  # against measured position. Without that board there is no "is", only a
  # "should be", and the bridge says so rather than reporting a guess.
  geometry = kinematics.load_chain_geometry()
  measures_tcp = any("joint_pos" in s.provides for s in sources)

  # Pay the first-inference cost here rather than inside the control loop: the
  # first forward pass allocates workspaces and warms up kernels, measured at
  # ~35 ms on cuda:0 vs ~2 ms on cpu. Inside the loop that would blow through
  # several 20 ms slots at exactly the moment the rig starts taking commands.
  warmup_ts = time.monotonic()
  with torch.inference_mode():
    policy({"actor": torch.zeros(1, layout.total_dim, dtype=torch.float32, device=device)})
  print(f"[INFO] Policy warm-up inference: {(time.monotonic() - warmup_ts) * 1000.0:.1f} ms")

  print(f"[INFO] Policy checkpoint: {checkpoint_path}")
  print(f"[INFO] Task: {args.task}")
  print(f"[INFO] Actor terms: {[t.name for t in layout.terms]} ({layout.total_dim} dim)")
  print(f"[INFO] Hardware sources: {[s.name for s in sources]}")
  print(f"[INFO] Control rate: {args.poll_hz:.1f} Hz")
  if any(s.name == "joints" for s in sources):
    # Print the sim joint order so a calibration bend can confirm the mapping.
    print(f"[INFO] joints: sim joint order (base->tip) = {list(layout.sim_joint_names)}")
    print(
      "[INFO] joints: sensor angles are fed base->tip (identity map). If a bent joint "
      "moves the wrong index, set --joint-order; if the sign is inverted, --joint-sign -1."
    )
  if measures_tcp:
    print(
      f"[INFO] Measured TCP: forward kinematics over the joint sensor, "
      f"{geometry.num_segments} segments, {geometry.total_length:.3f} m reach. "
      "The telemetry line shows soll/ist/err."
    )
  else:
    print(
      "[WARN] No joint sensor connected -- the rig cannot know where its tip is. "
      "Only the commanded target is reported; add --joint-port for the measured one."
    )
  print(
    f"[INFO] TCP target: ({args.target_x:+.3f}, {args.target_z:+.3f}) m in the base frame "
    "(x = sideways, z = up)"
  )
  print(
    "[INFO] Type a new target as 'x z' + Enter at any time, e.g. '0.20 0.37'. "
    f"Reachable shell: r = {workspace.SHELL_RADIUS} - {workspace.SHELL_CURVATURE}*angle^2 "
    f"(+-{workspace.SHELL_BAND} m), angle +-{workspace.ANGLE_LIMIT} rad from +z."
  )
  if args.dry_run:
    print("[INFO] --dry-run: computing actions but NOT sending start/f commands to the rig")
  target_ok = _announce_target(args.target_x, args.target_z, previous_ok=True, verbose=False)

  # Targets from the GUI arrive here. A separate process on a non-blocking
  # socket is the point: nothing the GUI does -- redraw, drag, stall, crash --
  # can consume any of the control loop's 20 ms budget.
  target_sock: socket.socket | None = None
  if args.target_port:
    try:
      target_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      target_sock.setblocking(False)
      target_sock.bind(("127.0.0.1", args.target_port))
      print(
        f"[INFO] Listening for GUI targets on udp://127.0.0.1:{args.target_port} "
        "(run: uv run python -m spirob_rl.rig.target_gui)"
      )
    except OSError as exc:
      # Not fatal: the rig can still be driven from the CLI and stdin.
      print(f"[WARN] Could not open target socket on port {args.target_port}: {exc}")
      target_sock = None

  # Open each source; each blocks only until its first reading is available.
  for source in sources:
    source.open()

  # Publish what the policy sees and does, for the target GUI. Opened after the
  # sources so a failure here cannot leave a rig half-initialised; it is a
  # measurement aid, never a precondition for driving the robot.
  publisher: telemetry.TelemetryPublisher | None = None
  if args.telemetry_port:
    try:
      publisher = telemetry.TelemetryPublisher(args.telemetry_host, args.telemetry_port)
      print(
        f"[INFO] Publishing tick telemetry to udp://{args.telemetry_host}:{args.telemetry_port} "
        "(the target GUI shows the measured tip from it)"
      )
    except OSError as exc:
      print(f"[WARN] Could not open telemetry socket: {exc}")

  assembler = observation.ObservationAssembler(layout, device)
  target = [args.target_x, args.target_z]
  target_seq = 0  # bumped on every target change; the observer segments reaches by it
  prev_raw_action = [0.0] * layout.action_dim

  poll_period_s = 1.0 / args.poll_hz
  monitor = LoopMonitor(period_s=poll_period_s)

  # Start the schedule only now: opening sources can take seconds, and anchoring
  # the grid before it would make the first ticks instantly overdue.
  next_tick = time.monotonic() + poll_period_s
  last_summary_ts = time.monotonic()
  last_print_ts = 0.0
  print_period_s = 1.0 / args.print_hz if args.print_hz > 0 else None
  stdin_open = True

  try:
    while True:
      # Sleep to the next slot. time.sleep can return early, so re-check
      # instead of trusting a single sleep to land on the deadline.
      while True:
        remaining = next_tick - time.monotonic()
        if remaining <= 0.0:
          break
        time.sleep(remaining)

      tick_start = monitor.begin()
      # Stay on the original grid: if a tick overran, skip the slots we missed
      # rather than shifting the phase forward (which would silently turn a
      # missed deadline into a permanently slower loop).
      next_tick += poll_period_s
      missed = 0
      while next_tick <= tick_start:
        next_tick += poll_period_s
        missed += 1
      monitor.overruns += missed

      # Non-blocking "x z" target update from stdin, if the user typed one.
      # select() reports a terminal ready only once a full line is buffered, so
      # readline() below cannot block mid-word and stall the control loop.
      ready, _, _ = select.select([sys.stdin], [], [], 0) if stdin_open else ((), (), ())
      if ready:
        line = sys.stdin.readline()
        if not line:
          # EOF (stdin closed or redirected from /dev/null): stop polling it,
          # otherwise select() reports ready forever and we spin every tick.
          stdin_open = False
        elif line.split():
          parsed = workspace.parse_target(line)
          if parsed is None:
            print(f"[WARN] Ignored input {line.strip()!r} -- expected 'x z' in metres")
          else:
            target = list(parsed)
            target_seq += 1
            target_ok = _announce_target(target[0], target[1], target_ok, verbose=True)

      # Targets from the GUI. Never blocks; only the newest datagram is used.
      if target_sock is not None:
        gui_target = _drain_target_socket(target_sock)
        if gui_target is not None and list(gui_target) != target:
          target = list(gui_target)
          target_seq += 1
          target_ok = _announce_target(target[0], target[1], target_ok, verbose=False)

      # Refresh every source, then read their signals (non-blocking).
      phase_ts = time.monotonic()
      merged_signals: dict[str, list[float]] = {}
      for source in sources:
        source.poll()
        merged_signals.update(source.signals())

      # Where the arm actually is, from the angles the board just reported.
      # Inside the drain phase on purpose: it is part of reading the sensors,
      # and this way its cost shows up in the [drain] number like everything
      # else rather than hiding between the measured phases.
      tcp_actual: tuple[float, float] | None = None
      if measures_tcp and "joint_pos" in merged_signals:
        tcp_actual = kinematics.tcp_position(merged_signals["joint_pos"], geometry)
      drain_ms = (time.monotonic() - phase_ts) * 1000.0

      obs = assembler.assemble(target, prev_raw_action, merged_signals)

      phase_ts = time.monotonic()
      with torch.inference_mode():
        action = policy({"actor": obs})
      raw_action = action.reshape(-1).tolist()
      infer_ms = (time.monotonic() - phase_ts) * 1000.0

      phase_ts = time.monotonic()
      force_setpoints_n = motor.send_action(raw_action)
      tx_ms = (time.monotonic() - phase_ts) * 1000.0

      # Telemetry last: the observer wants the finished tick, and anything that
      # goes wrong here must land after the rig already has its command.
      tele_ms = 0.0
      if publisher is not None:
        phase_ts = time.monotonic()
        publisher.send(
          telemetry.BridgeState(
            t=tick_start,
            tick=monitor.ticks,
            target=(target[0], target[1]),
            target_seq=target_seq,
            tcp_actual=tcp_actual,
            raw_action=raw_action,
            force_n=force_setpoints_n,
            signals=merged_signals,
            dry_run=args.dry_run,
            task=args.task,
          )
        )
        tele_ms = (time.monotonic() - phase_ts) * 1000.0

      # Always report a missed deadline, even when the line is throttled --
      # throttling is for readability and must not hide dropped control steps.
      due = print_period_s is not None and (tick_start - last_print_ts) >= print_period_s
      if due or missed:
        busy_ms = drain_ms + infer_ms + tx_ms + tele_ms
        overrun_marker = f" OVERRUN(+{missed})" if missed else ""
        source_debug = " | ".join(s.debug() for s in sources)
        print(
          f"dt={monitor.last_period_ms:6.2f}ms ({1000.0 / monitor.last_period_ms:5.1f}Hz) "
          f"busy={busy_ms:5.2f}ms [drain {drain_ms:4.2f}|infer {infer_ms:4.2f}|tx {tx_ms:4.2f}"
          f"|tele {tele_ms:4.2f}]"
          f"{overrun_marker} | {_format_tcp(target, tcp_actual)} | "
          f"{source_debug} | raw_action={[f'{a:.3f}' for a in raw_action]} -> "
          f"force_n={[f'{f:.2f}' for f in force_setpoints_n]}",
        )
        last_print_ts = tick_start

      if time.monotonic() - last_summary_ts >= _SUMMARY_INTERVAL_S:
        print(monitor.summary())
        last_summary_ts = time.monotonic()

      # Closed last so the tick's own console output counts as work: printing
      # is real time spent between two policy evaluations.
      monitor.end()

      prev_raw_action = raw_action
  except KeyboardInterrupt:
    pass
  finally:
    print(monitor.summary())
    if publisher is not None:
      # Errors here are almost always "no observer running", which is fine --
      # worth one line at the end, never one per tick.
      print(f"[LOOP] telemetry: {publisher.sends} datagrams sent, {publisher.errors} failed")
      publisher.close()
    for source in sources:
      try:
        source.close()
      except Exception:
        pass


if __name__ == "__main__":
  main()
