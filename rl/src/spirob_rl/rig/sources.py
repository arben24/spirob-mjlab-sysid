"""Hardware sources for the SpiRob policy bridge.

A *source* is one physical device -- one USB port, one protocol -- that exposes
a set of named signal vectors and, optionally, accepts commands. The bridge
never talks to a serial port directly; it polls sources and reads their
``signals()``. This is the seam that lets one bridge drive many hardware
configurations: add a sensor by writing a source and registering which
observation terms its signals feed (see ``observation.py``); nothing else in
the loop changes.

Every source is single-threaded and non-blocking on ``poll()``. That is
deliberate and load-bearing: the control loop's 50 Hz schedule was validated
against exactly this discipline, and a source that blocked (or ran a helper
thread fighting for the GIL) would eat into the 20 ms tick budget.

Two sources exist today:

* ``MotorRig`` -- the tendon motor MCU (``pio_project/``, 460800 baud). Provides
  the force/tendon signals and is also the actuator: it receives the policy's
  force setpoints. Mandatory for every realizable task.
* ``JointSensor`` -- the accelerometer board (``rig/acc_board/``, 1 Mbaud on
  a second USB port). Turns raw acceleration frames into the 13 joint angles
  and each segment's inclination, which makes both the "joints" and the "imu"
  sensor level realizable -- and, through the forward kinematics, is also the
  rig's only way to know where its own tip is.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np
import serial

from .serial_protocol import (
  BAUD_RATE as MOTOR_BAUD_RATE,
  NUM_ACTUATORS,
  Status,
  drain_latest_status,
  open_serial_port,
  send_command,
)
from .kinematics import segment_pitch_cos_sin
from .acc_board import frame_parser as _fp
from .acc_board.kinematic_optimization import SpirobKinematics


@runtime_checkable
class HardwareSource(Protocol):
  """One physical device exposing named signal vectors."""

  name: str
  provides: tuple[str, ...]

  def open(self) -> None:
    """Connect and block only until the first reading is available."""
    ...

  def poll(self) -> None:
    """Refresh internal state from the device. Must not block."""
    ...

  def signals(self) -> dict[str, list[float]]:
    """Latest value of each provided signal, keyed by signal name."""
    ...

  def debug(self) -> str:
    """One-line human-readable state for the telemetry log."""
    ...

  def close(self) -> None:
    ...


class _FilteredDiff:
  """Finite-difference velocity with one-pole smoothing.

  Shared by every source that reports a velocity: low delay, but far less
  derivative noise than a raw difference. Same filter the pendulum bridge
  used for its motor/pendulum velocities.
  """

  def __init__(self, size: int, alpha: float = 0.15) -> None:
    self.alpha = alpha
    self.vel = [0.0] * size
    self._prev: list[float] | None = None

  def update(self, values: list[float], dt: float) -> list[float]:
    if self._prev is not None and dt > 0.0:
      for i, value in enumerate(values):
        raw = (value - self._prev[i]) / dt
        self.vel[i] = self.alpha * raw + (1.0 - self.alpha) * self.vel[i]
    self._prev = list(values)
    return list(self.vel)


def raw_action_to_force_n(raw_action: list[float]) -> list[float]:
  """Map a policy action to per-tendon force setpoints [N] for the motor MCU.

  Replicates ``TendonEffortActionCfg(scale=75, offset=-75, clip=(-150, 0))``
  from ``reach_env_cfg.py`` and flips the sign: the sim ctrl is a pulling
  (<= 0 N) effort, ``clip(raw*75 - 75, -150, 0)``; the firmware's ``f`` command
  wants a positive tension magnitude, i.e. its negation ``clip(75 - raw*75, 0,
  150)``. Action-space knowledge, so it lives with the actuator source.
  """
  return [max(0.0, min(150.0, 75.0 - a * 75.0)) for a in raw_action]


class MotorRig:
  """Tendon motor MCU: force/tendon telemetry in, force setpoints out.

  Also the actuator for every task, so it is always required. ``rope_len_mm``
  from the firmware is cumulative spool winding, zeroed at boot / ``n`` /
  homing; negating and scaling it to metres gives the sim's
  ``tendon_len_rel``. Sign and zero-reference caveats are documented on the
  bridge's ``--tendon-sign`` / ``--null-tendons`` options.
  """

  name = "motor"
  provides = ("tendon_len", "tendon_vel", "force")

  def __init__(
    self,
    port: str,
    baudrate: int = MOTOR_BAUD_RATE,
    tendon_sign: tuple[float, ...] = (1.0, 1.0),
    dry_run: bool = False,
    null_on_start: bool = False,
    vel_alpha: float = 0.15,
  ) -> None:
    if len(tendon_sign) != NUM_ACTUATORS:
      raise ValueError(f"tendon_sign needs {NUM_ACTUATORS} values, got {tendon_sign!r}")
    self.port = port
    self.baudrate = baudrate
    self.tendon_sign = tendon_sign
    self.dry_run = dry_run
    self.null_on_start = null_on_start
    self._ser: serial.Serial | None = None
    self._status: Status | None = None
    self._prev_status: Status | None = None
    self._tendon_len = [0.0] * NUM_ACTUATORS
    self._diff = _FilteredDiff(NUM_ACTUATORS, vel_alpha)

  def open(self) -> None:
    self._ser = open_serial_port(self.port, self.baudrate)
    if self.null_on_start:
      print("[WARN] Nulling rope length now -- make sure the arm is physically straight!")
      send_command(self._ser, "n all")
      time.sleep(0.05)
    if not self.dry_run:
      send_command(self._ser, "start all")
      print("[INFO] motor: force control started (start all)")
    # Block only for the first packet, so downstream code always has a reading.
    wait_start = time.monotonic()
    while self._status is None:
      self.poll()
      if self._status is None:
        if time.monotonic() - wait_start > 5.0:
          raise SystemExit(
            f"motor: no telemetry on {self.port} after 5s -- check port/baud/wiring."
          )
        time.sleep(0.01)
    print(
      f"[INFO] motor: first telemetry force={self._status.force_n} "
      f"rope_mm={self._status.rope_len_mm}"
    )

  def poll(self) -> None:
    assert self._ser is not None
    latest = drain_latest_status(self._ser)
    if latest is None:
      return  # keep last known state; the loop tolerates a stale tick
    self._prev_status = self._status
    self._status = latest

    dt = 0.02
    if self._prev_status is not None and latest.timestamp_us > self._prev_status.timestamp_us:
      dt = max((latest.timestamp_us - self._prev_status.timestamp_us) * 1e-6, 1e-3)

    # rope_len_mm zeroed at the null reference, so -rope_len_mm/1000 is the
    # tendon_len_rel analogue. Sign flips it per motor if the spool convention
    # is reversed (see --tendon-sign).
    self._tendon_len = [
      self.tendon_sign[i] * (-latest.rope_len_mm[i] / 1000.0) for i in range(NUM_ACTUATORS)
    ]
    self._diff.update(self._tendon_len, dt)

  def signals(self) -> dict[str, list[float]]:
    force = list(self._status.force_n) if self._status is not None else [0.0] * NUM_ACTUATORS
    return {
      "tendon_len": list(self._tendon_len),
      "tendon_vel": list(self._diff.vel),
      "force": force,
    }

  def send_action(self, raw_action: list[float]) -> list[float]:
    """Translate a policy action to force setpoints and transmit them.

    Returns the force setpoints [N] for logging. A no-op sender in dry-run.
    """
    return self.send_forces(raw_action_to_force_n(raw_action))

  def send_forces(self, forces_n: list[float]) -> list[float]:
    """Transmit explicit per-tendon force setpoints [N]. No-op in dry-run.

    Used by the online-training env, which caps forces for safety before
    sending, so the mapping cannot be folded into ``send_action``.
    """
    if not self.dry_run and self._ser is not None:
      for i in range(NUM_ACTUATORS):
        send_command(self._ser, f"f {i} {forces_n[i]:.2f}")
    return forces_n

  def debug(self) -> str:
    if self._status is None:
      return "motor: <no data>"
    return (
      f"force={[f'{f:.2f}' for f in self._status.force_n]} "
      f"rope_mm={[f'{r:.1f}' for r in self._status.rope_len_mm]} "
      f"tendon_len={[f'{v:.4f}' for v in self._tendon_len]}"
    )

  def close(self) -> None:
    if self._ser is not None:
      try:
        send_command(self._ser, "stop")
        print("[INFO] motor: stopped")
      except Exception:
        pass
      self._ser.close()


class JointSensor:
  """Accelerometer board -> 13 joint angles, their velocities, segment pitch.

  Wraps ``rig/acc_board``: reads the binary acceleration frames, EMA-
  smooths per sensor (as the reference ``SerialReader`` does), and reads the
  joint angles in closed form via ``solve_angles_direct`` (~0.005 ms, invisible
  in the 20 ms budget).

  The board is the rig's only source of shape, and it feeds two different
  sensor levels:

  * ``joints`` gets ``joint_pos``/``joint_vel`` -- the angles themselves.
  * ``imu`` gets ``segment_pitch`` -- each segment's absolute inclination as
    cos/sin. That is what the accelerometers physically measure;
    ``solve_angles_direct`` reads the inclinations and returns their
    differences, and ``kinematics.segment_pitch_cos_sin`` sums them back up.
    The gravity reference and any tilt of the whole mount cancel in that round
    trip, so this needs no calibration beyond the sign and order below.

  Two hardware-alignment parameters this code cannot verify from the sim side
  -- confirm on the first run:

  * Order: the solver returns angles base->tip, and the sim's actor joint order
    (``j_12 .. j_0``) is also base->tip, so the default is identity. Only the
    *segment numbering* differs between the two conventions, not the array
    order. Override with ``order`` (a 13-index permutation) if a calibration
    bend shows the wrong index moving.
  * Sign: which bending direction is positive is a mounting detail. Flip with
    ``sign=-1.0`` if the arm's measured angles oppose the sim's.
  """

  name = "joints"
  provides = ("joint_pos", "joint_vel", "segment_pitch")

  # Sensor board IDs in physical base->tip order (from the module README).
  SENSOR_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37)

  def __init__(
    self,
    port: str,
    baudrate: int = _fp.BAUD_RATE,
    sign: float = 1.0,
    order: tuple[int, ...] | None = None,
    ema_alpha: float = 0.25,
    vel_alpha: float = 0.15,
  ) -> None:
    self.port = port
    self.baudrate = baudrate
    self.sign = sign
    self.ema_alpha = ema_alpha
    self._kin = SpirobKinematics(sensor_id_mapping=list(self.SENSOR_IDS), rotation_axis="x")
    self.num_joints = self._kin.num_joints  # 13
    if order is None:
      self.order = tuple(range(self.num_joints))  # identity: both are base->tip
    else:
      if sorted(order) != list(range(self.num_joints)):
        raise ValueError(
          f"joint order must be a permutation of 0..{self.num_joints - 1}, got {order!r}"
        )
      self.order = tuple(order)
    self._parser = _fp.BinaryFrameParser()
    self._smooth_acc: dict[int, np.ndarray] = {}
    self._angles = [0.0] * self.num_joints
    self._diff = _FilteredDiff(self.num_joints, vel_alpha)
    self._last_poll_ts: float | None = None
    self._ser: serial.Serial | None = None
    self._have_frame = False

  def open(self) -> None:
    self._ser = serial.Serial(self.port, self.baudrate, timeout=0.001)
    time.sleep(0.1)
    self._ser.reset_input_buffer()
    self._ser.write(b"a")  # ACC_ONLY mode: highest frame rate, we only use acc
    wait_start = time.monotonic()
    while not self._have_frame:
      self.poll()
      if not self._have_frame:
        if time.monotonic() - wait_start > 5.0:
          raise SystemExit(
            f"joints: no sensor frames on {self.port} after 5s -- check port/baud/wiring."
          )
        time.sleep(0.01)
    print(f"[INFO] joints: first angles (deg) {[round(a, 1) for a in self._angles_deg()]}")

  def poll(self) -> None:
    assert self._ser is not None
    waiting = self._ser.in_waiting
    if waiting:
      self._parser.add_data(self._ser.read(waiting))
    frames = self._parser.extract_frames()
    if not frames:
      return
    # Newest complete frame wins, like the reference SerialReader.
    sensors = frames[-1]["sensors"]
    for sid, vals in sensors.items():
      raw = np.array(vals[:3], dtype=float)
      if sid not in self._smooth_acc:
        self._smooth_acc[sid] = raw
      else:
        self._smooth_acc[sid] = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._smooth_acc[sid]

    # Feed a full mapping so prepare_measured_data never warns about a gap:
    # a sensor missing from this frame keeps its last smoothed value.
    acc_dict = {sid: self._smooth_acc[sid].tolist() for sid in self._smooth_acc}
    if not all(sid in acc_dict for sid in self.SENSOR_IDS):
      return  # not every sensor seen yet; wait rather than feed [0,0,1] holes
    prepared = self._kin.prepare_measured_data(acc_dict)
    q = self._kin.solve_angles_direct(prepared)  # radians, base->tip
    reordered = [self.sign * float(q[self.order[j]]) for j in range(self.num_joints)]
    self._angles = reordered

    now = time.monotonic()
    dt = 0.02 if self._last_poll_ts is None else max(now - self._last_poll_ts, 1e-3)
    self._last_poll_ts = now
    self._diff.update(self._angles, dt)
    self._have_frame = True

  def _angles_deg(self) -> list[float]:
    return [a * 180.0 / np.pi for a in self._angles]

  def signals(self) -> dict[str, list[float]]:
    return {
      "joint_pos": list(self._angles),
      "joint_vel": list(self._diff.vel),
      "segment_pitch": segment_pitch_cos_sin(self._angles),
    }

  def debug(self) -> str:
    return f"joints_deg={[f'{a:+.1f}' for a in self._angles_deg()]}"

  def close(self) -> None:
    if self._ser is not None:
      self._ser.close()
