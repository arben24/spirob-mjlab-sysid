#!/usr/bin/env python3
"""
SpiRob Digital Twin – MuJoCo ↔ Hardware Bridge

Opens the MuJoCo viewer. When you drag tendon actuator sliders in the GUI,
the resulting force setpoints (Newtons) are forwarded to the real ESP32
hardware via serial force-control commands.

Requires the `one_motor_system_id` firmware (460800 baud, binary telemetry).
"""

import struct
import time

import mujoco as mj
import serial

from spirob.paths import IDENTIFIED_MODEL

# ── Serial config (same as live_monitor_simple.py) ──────────────────────────
PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
HEADER = b"\xaa\x55"
STEP_END = b"\xbb\x66"
STRUCT_FMT = "<I ff ff"  # uint32 ts_us, float force[2], float rope_mm[2]
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)  # 20

SEND_HZ = 50  # max command rate to ESP32
SEND_INTERVAL = 1.0 / SEND_HZ
FORCE_DEADBAND = 0.05  # N – ignore changes smaller than this


# ── Helpers ──────────────────────────────────────────────────────────────────

def send_cmd(ser: serial.Serial, cmd: str) -> None:
    """Send an ASCII command (auto-appends newline)."""
    ser.write((cmd + "\n").encode("ascii"))


def drain_telemetry(ser: serial.Serial):
    """Read all pending binary packets; return latest status tuple or None."""
    latest = None
    while ser.in_waiting >= 2 + STRUCT_SIZE:
        b0 = ser.read(1)
        if b0 == b"\xaa":
            b1 = ser.read(1)
            if b1 == b"\x55":
                pkt = ser.read(STRUCT_SIZE)
                if len(pkt) == STRUCT_SIZE:
                    latest = struct.unpack(STRUCT_FMT, pkt)
        elif b0 == b"\xbb":
            ser.read(2)  # step-end: marker byte + motor index
    return latest


# ── MuJoCo model setup ──────────────────────────────────────────────────────
xml_path = IDENTIFIED_MODEL

model = mj.MjModel.from_xml_path(str(xml_path))

# cylinder = spec.worldbody.add_body(name="cylinder", pos=[-0.11, 0.00, 0.11+0.053])
# cylinder.add_geom(
#     name="cyl_geom",
#     type=mj.mjtGeom.mjGEOM_CYLINDER,
#     size=[0.05, 0.15, 0.05],
#     euler=[90, 0, 0],
#     rgba=[0.2, 0.8, 0.5, 1],
#     density=1000,
# )

# spirob = spec.body("seg_0")
#model = spec.compile()
data = mj.MjData(model)
print("MuJoCo model loaded.")

data.ctrl[0] = -5.0
data.ctrl[1] = -5.0


# ── Serial connection ───────────────────────────────────────────────────────

ser = serial.Serial(PORT, BAUDRATE, timeout=0.001)
time.sleep(0.1)
ser.reset_input_buffer()
print(f"Verbunden: {PORT} @ {BAUDRATE}")

send_cmd(ser, "f all 5")
send_cmd(ser, "start all")
print("Kraftregelung gestartet (start all)")


# ── Main loop ───────────────────────────────────────────────────────────────

prev_f = [None, None]
last_send_t = 0.0

try:
    with mj.viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 360:
            step_start = time.time()

            # 1) Read actuator ctrl values from GUI (Newtons)
            f0 = -1 * float(data.ctrl[0])
            f1 = -1 * float(data.ctrl[1])

            # 2) Forward to hardware (throttled, with dead-band)
            now = time.time()
            if now - last_send_t >= SEND_INTERVAL:
                if prev_f[0] is None or abs(f0 - prev_f[0]) > FORCE_DEADBAND:
                    send_cmd(ser, f"f 0 {f0:.2f}")
                    prev_f[0] = f0
                if prev_f[1] is None or abs(f1 - prev_f[1]) > FORCE_DEADBAND:
                    send_cmd(ser, f"f 1 {f1:.2f}")
                    prev_f[1] = f1
                last_send_t = now

            # 3) Drain hardware telemetry (prevents serial buffer overflow)
            hw = drain_telemetry(ser)
            if hw:
                _ts, hf0, hf1, hr0, hr1 = hw
                print(
                    f"\rHW: {hf0:6.2f}N {hf1:6.2f}N | "
                    f"{hr0:7.1f}mm {hr1:7.1f}mm  ",
                    end="",
                    flush=True,
                )

            # 4) Step simulation & sync viewer
            mj.mj_step(model, data)
            v.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

finally:
    send_cmd(ser, "stop")
    print("\nMotoren gestoppt.")
    ser.close()
    print("Serielle Verbindung geschlossen.")





