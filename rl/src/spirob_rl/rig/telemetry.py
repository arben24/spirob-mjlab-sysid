"""Bridge-to-observer telemetry: one datagram per control tick.

The policy bridge is the only process that knows what the policy was told, what
it decided, and -- via the accelerometer board and the forward kinematics --
where the arm actually ended up. Anything that wants to watch needs that
stream: today ``target_gui.py``, which draws the commanded target and the
measured tip next to each other. This module defines the wire format once so
both ends cannot drift apart, in the same spirit as
``COMMUNICATION_PROTOCOL.md`` for the motor MCU.

Design choices, all for the same reason (the control loop must not notice):

* **UDP, connected socket, non-blocking, errors swallowed.** Sending never
  waits for a receiver, applies no backpressure, and a missing, slow or dead
  observer cannot stall a tick. A lost datagram costs one sample of a
  measurement stream; blocking would cost a control step.
* **One packet per tick, no buffering.** The observer aligns by timestamp, so
  it wants the raw stream, not a smoothed summary.
* **JSON lines.** Measured: a full packet for a joints policy is 382 bytes and
  costs 13 us to build and send (35 us worst case over 20 000 sends), i.e.
  0.2 % of a 20 ms tick -- and ``nc -ul 47801`` shows you what is going out.
  Should that ever get tight, this file is the only place that has to change.

``t`` is ``time.monotonic()`` from the bridge process, so an observer on the
same machine can compare timestamps directly.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field

TELEMETRY_UDP_PORT = 47801
"""Default localhost UDP port the bridge publishes its per-tick state on.

Deliberately one above ``workspace.TARGET_UDP_PORT`` (47800), which carries
setpoints in the other direction.
"""

_FLOAT_DIGITS = 6


def _round(values) -> list[float]:
  return [round(float(v), _FLOAT_DIGITS) for v in values]


@dataclass
class BridgeState:
  """What the bridge knew and did during one control tick."""

  t: float
  """``time.monotonic()`` at the start of the tick."""
  tick: int
  target: tuple[float, float]
  """Commanded TCP target (x, z) in metres, base frame -- where it *should* be."""
  target_seq: int
  """Increments on every target change -- lets an observer segment reaches."""
  tcp_actual: tuple[float, float] | None = None
  """Measured TCP (x, z) in metres -- where it *is*.

  Forward kinematics over the accelerometer board's joint angles. None whenever
  that board is not connected, because then nothing on the rig knows.
  """
  raw_action: list[float] = field(default_factory=list)
  force_n: list[float] = field(default_factory=list)
  signals: dict[str, list[float]] = field(default_factory=dict)
  """Named hardware signals as the sources reported them (tendon_len, joint_pos, ...)."""
  dry_run: bool = False
  task: str = ""

  def encode(self) -> bytes:
    payload = {
      "t": round(self.t, _FLOAT_DIGITS),
      "tick": self.tick,
      "target": _round(self.target),
      "target_seq": self.target_seq,
      "tcp_actual": _round(self.tcp_actual) if self.tcp_actual is not None else None,
      "raw_action": _round(self.raw_action),
      "force_n": _round(self.force_n),
      "signals": {name: _round(values) for name, values in self.signals.items()},
      "dry_run": self.dry_run,
      "task": self.task,
    }
    return json.dumps(payload, separators=(",", ":")).encode("ascii")

  @classmethod
  def decode(cls, data: bytes) -> BridgeState | None:
    """Parse a datagram, or None if it is not one of ours.

    Never raises: a stray packet on the port must not take the observer down.
    """
    try:
      payload = json.loads(data.decode("ascii"))
      actual = payload.get("tcp_actual")
      return cls(
        t=float(payload["t"]),
        tick=int(payload["tick"]),
        target=(float(payload["target"][0]), float(payload["target"][1])),
        target_seq=int(payload["target_seq"]),
        tcp_actual=(float(actual[0]), float(actual[1])) if actual else None,
        raw_action=[float(v) for v in payload.get("raw_action", [])],
        force_n=[float(v) for v in payload.get("force_n", [])],
        signals={k: [float(x) for x in v] for k, v in payload.get("signals", {}).items()},
        dry_run=bool(payload.get("dry_run", False)),
        task=str(payload.get("task", "")),
      )
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
      return None


class TelemetryPublisher:
  """Fire-and-forget sender for the bridge side.

  ``send`` is the only method the control loop calls, and it can neither block
  nor raise. Failures are counted so the bridge can mention them once instead
  of printing per tick.
  """

  def __init__(self, host: str = "127.0.0.1", port: int = TELEMETRY_UDP_PORT) -> None:
    self.address = (host, port)
    self.sends = 0
    self.errors = 0
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.setblocking(False)
    # Connect a datagram socket: sendto's per-call route lookup goes away and
    # ICMP "port unreachable" from an absent observer lands as an error we
    # count rather than an exception at a random later send.
    self._sock.connect(self.address)

  def send(self, state: BridgeState) -> None:
    try:
      self._sock.send(state.encode())
      self.sends += 1
    except OSError:
      self.errors += 1

  def close(self) -> None:
    self._sock.close()


class TelemetryReceiver:
  """Observer side: bind, drain, never block."""

  def __init__(self, port: int = TELEMETRY_UDP_PORT, host: str = "127.0.0.1") -> None:
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._sock.setblocking(False)
    self._sock.bind((host, port))
    self.received = 0
    self.malformed = 0

  def drain(self, limit: int = 512) -> list[BridgeState]:
    """Every datagram that arrived since the last call, oldest first."""
    states: list[BridgeState] = []
    for _ in range(limit):
      try:
        data = self._sock.recv(8192)
      except (BlockingIOError, InterruptedError):
        break
      except OSError:
        break
      state = BridgeState.decode(data)
      if state is None:
        self.malformed += 1
      else:
        self.received += 1
        states.append(state)
    return states

  def close(self) -> None:
    self._sock.close()
