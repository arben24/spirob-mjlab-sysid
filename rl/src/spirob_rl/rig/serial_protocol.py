"""Serial protocol shared with the SpiRob ESP32 firmware (`pio_project/`).

Counterpart to `inverted_pendulum/telemetry_viser.py`, but for the SpiRob
two-motor tendon rig. See `../COMMUNICATION_PROTOCOL.md` for the full spec;
this module only implements the pieces `policy_bridge.py` needs.

Wire format (460800 baud):

* Host -> MCU: ASCII commands, newline-terminated (e.g. ``b"f 0 42.50\\n"``).
* MCU -> Host: binary status packets, pushed continuously and unsolicited
  (~600 Hz), no request needed:

    0xAA 0x55 | uint32 timestamp_us | float force_0 | float force_1
              | float rope_len_0_mm | float rope_len_1_mm

  22 bytes total (2-byte header + 20-byte payload). There is also a 3-byte
  "step end" packet (``0xBB 0x66`` + motor index) emitted by the firmware's
  blocking `step`/`c`/`r` commands; the policy bridge never issues those
  commands, but the parser skips step-end packets if it sees one so a stray
  packet doesn't desync the reader.
"""

from __future__ import annotations

import struct
import time
from typing import NamedTuple, Optional

import serial
from serial.tools import list_ports

BAUD_RATE = 460800
NUM_ACTUATORS = 2

HEADER = b"\xaa\x55"
STEP_END_HEADER = b"\xbb\x66"

STATUS_STRUCT = struct.Struct("<Iff ff")  # ts_us, force[2], rope_len_mm[2]
STATUS_STRUCT_SIZE = STATUS_STRUCT.size  # 20


class Status(NamedTuple):
  timestamp_us: int
  force_n: tuple[float, float]
  rope_len_mm: tuple[float, float]


def find_default_port() -> Optional[str]:
  ports = list(list_ports.comports())
  if not ports:
    return None
  for port in ports:
    description = f"{port.device} {port.description}".lower()
    if any(keyword in description for keyword in ("usb", "serial", "com")):
      return port.device
  return ports[0].device


def open_serial_port(port_name: str, baudrate: int = BAUD_RATE) -> serial.Serial:
  ser = serial.Serial(port=port_name, baudrate=baudrate, timeout=0.001)
  time.sleep(0.1)
  ser.reset_input_buffer()
  return ser


def send_command(ser: serial.Serial, command: str) -> None:
  """Send an ASCII command line (auto-appends the newline terminator)."""
  ser.write((command if command.endswith("\n") else command + "\n").encode("ascii"))


def _parse_status(payload: bytes) -> Status:
  ts_us, f0, f1, r0, r1 = STATUS_STRUCT.unpack(payload)
  return Status(timestamp_us=ts_us, force_n=(f0, f1), rope_len_mm=(r0, r1))


def drain_latest_status(ser: serial.Serial) -> Optional[Status]:
  """Read and discard all buffered packets, returning the newest status.

  The firmware streams telemetry continuously and much faster (~600 Hz) than
  the policy runs (50 Hz), so each control tick just wants "whatever the rig
  looks like right now" -- draining the buffer and keeping only the latest
  status packet is both simplest and lowest-latency. Malformed bytes (partial
  packets, noise) are skipped byte-by-byte until the next header is found.
  """
  latest: Optional[Status] = None
  while ser.in_waiting >= 1:
    byte0 = ser.read(1)
    if not byte0:
      break
    if byte0 == HEADER[:1]:
      byte1 = ser.read(1)
      if byte1 != HEADER[1:]:
        continue
      payload = ser.read(STATUS_STRUCT_SIZE)
      if len(payload) != STATUS_STRUCT_SIZE:
        break
      latest = _parse_status(payload)
    elif byte0 == STEP_END_HEADER[:1]:
      byte1 = ser.read(1)
      if byte1 != STEP_END_HEADER[1:]:
        continue
      ser.read(1)  # motor index byte, not needed here
    # else: stray byte (e.g. a human-readable Serial.printf line), skip it.
  return latest
