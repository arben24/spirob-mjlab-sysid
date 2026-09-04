#!/usr/bin/env python3
"""
SpiRob auto-recording with predefined tendon-force trajectories

Drives the motors automatically through a predefined
sequence of phases (constant, ramp, sine). At the same time the ArUco sync
marker is displayed and all data (including
Hardware-Metriken) in ein CSV-Datenframe geschrieben.
"""

import struct
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
import pandas as pd
import serial

from spirob.paths import DEFAULT_MODEL
from spirob.paths import build_dir as _build_dir

# =============================================================================
# ── 1. FORCE PROFILE DEFINITIONS ─────────────────────────────────────────────
# =============================================================================

class ForceProfile(ABC):
    @abstractmethod
    def get_force(self, t: float, duration: float) -> float:
        """Return the target force at local time t (0 <= t <= duration)."""
        pass

class Constant(ForceProfile):
    def __init__(self, force: float):
        self.force = force
    def get_force(self, t: float, duration: float) -> float:
        return self.force

class Ramp(ForceProfile):
    def __init__(self, start_f: float, end_f: float):
        self.start_f = start_f
        self.end_f = end_f
    def get_force(self, t: float, duration: float) -> float:
        progress = np.clip(t / duration, 0.0, 1.0)
        return self.start_f + progress * (self.end_f - self.start_f)

class Sine(ForceProfile):
    def __init__(self, offset: float, amplitude: float, period: float, min_force: float = 10.0):
        self.offset = offset
        self.amplitude = amplitude
        self.period = period
        self.min_force = min_force
    def get_force(self, t: float, duration: float) -> float:
        val = self.offset + self.amplitude * np.sin(2 * np.pi * t / self.period)
        return max(self.min_force, val)

class Phase:
    def __init__(self, duration: float, m0: ForceProfile, m1: ForceProfile):
        self.duration = duration
        self.m0 = m0
        self.m1 = m1

# =============================================================================
# ── 2. SEQUENCE SETTINGS (EDIT THESE) ────────────────────────────────────────
# =============================================================================
# Define here which forces are commanded, one phase after another.
# Motor 0 = m0, Motor 1 = m1.

max_force = 80.0  # maximum force [N]
min_force = 10.0   # minimum force [N]
offset = 50.0
amplitude = 40.0
period = 10.0

# PHASES = [
#     # 1. Pre-tension: hold both tendons gently at 2 N for 5 seconds
#     Phase(duration=5.0, m0=Constant(min_force), m1=Constant(min_force)),
    
#     # 2. Ramp M0: motor 1 fixed (5 N), motor 0 ramps 5 N -> 50 N over 5 s
#     Phase(duration=10.0, m0=Ramp(min_force, max_force), m1=Constant(min_force)),

#     Phase(duration=5.0, m0=Ramp(max_force, min_force), m1=Constant(min_force)),
    
#     # 3. Ramp M1: motor 0 fixed (50 N), motor 1 follows (5 N -> 50 N over 5 s)
#     Phase(duration=10.0, m0=Constant(min_force), m1=Ramp(min_force, max_force)),

#     Phase(duration=5.0, m0=Constant(min_force), m1=Ramp(max_force, min_force)),
    
#     # 4. Sine test: M1 fixed (50 N), M0 sine (centre 50 N, +/- 4 N, 2 s period) for 10 s
#     Phase(duration=10.0, m0=Ramp(min_force, offset), m1=Constant(min_force)),
#     Phase(duration=10.0, m0=Sine(offset=offset, amplitude=amplitude, period=period), m1=Constant(min_force)),
#     Phase(duration=5.0, m0=Ramp(offset, min_force), m1=Constant(min_force)),


#     Phase(duration=10.0, m0=Constant(min_force), m1=Ramp(min_force, offset)),
#     Phase(duration=10.0, m0=Constant(min_force), m1=Sine(offset=offset, amplitude=amplitude, period=period)),
#     Phase(duration=5.0, m0=Constant(min_force), m1=Ramp(offset, min_force)),
#     # 5. Controlled release: both motors together, 10 N -> 0 N over 5 s
#     #Phase(duration=5.0, m0=Ramp(10.0, 0.0), m1=Ramp(10.0, 0.0)),
# ]

# PHASES = [

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(80), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(100), m1=Constant(10)),

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(80)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(10), m1=Constant(100)),

#     Phase(duration=5.0, m0=Constant(10), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(80), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(100), m1=Constant(30)),

#     Phase(duration=5.0, m0=Constant(30), m1=Constant(10)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(80)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(30)),
#     Phase(duration=5.0, m0=Constant(30), m1=Constant(100)),
    
# ]

# PHASES = [

#     Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

#     Phase(duration=10.0, m0=Sine(offset=30, amplitude=80, period=10.0), m1=Constant(10)),
#     Phase(duration=10.0, m0=Constant(10), m1=Sine(offset=30, amplitude=80, period=10.0)),
#     Phase(duration=10.0, m0=Sine(offset=30, amplitude=80, period=10.0), m1=Constant(30)),
#     Phase(duration=5.0, m0=Sine(offset=30, amplitude=30, period=10.0), m1=Sine(offset=50, amplitude=50, period=5.0)),
#     Phase(duration=5.0, m0=Sine(offset=60, amplitude=30, period=5.0), m1=Sine(offset=50, amplitude=50, period=10.0)),
#     Phase(duration=15.0, m0=Sine(offset=60, amplitude=50, period=5.0), m1=Sine(offset=50, amplitude=50, period=10.0)),
#     #Phase(duration=5.0, m0=Constant(10), m1=Sine(offset=50, amplitude=30, period=2.0)),
#     #Phase(duration=5.0, m0=Constant(30), m1=Sine(offset=50, amplitude=30, period=2.0)),

#     Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    
# ]


PHASES = [

    Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    Phase(duration=5.0, m0=Constant(10), m1=Ramp(10, 170)),
    Phase(duration=5.0, m0=Constant(10), m1=Ramp(170, 10)),

    Phase(duration=5.0, m0=Ramp(10, 170), m1=Constant(10)),
    Phase(duration=5.0, m0=Ramp(170, 10), m1=Constant(10)),

    Phase(duration=5.0, m0=Constant(40), m1=Ramp(10, 150)),
    Phase(duration=5.0, m0=Constant(40), m1=Ramp(150, 10)),

    Phase(duration=5.0, m0=Ramp(10, 170), m1=Constant(40)),
    Phase(duration=5.0, m0=Ramp(170, 10), m1=Constant(40)),

    Phase(duration=1.0, m0=Constant(10), m1=Constant(10)),

    
]


# =============================================================================
# ── INTERNE KONFIGURATION (SERIELL / MUJOCO / ARUCO) ─────────────────────────
# =============================================================================

PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
STRUCT_FMT = "<I ff ff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

SEND_HZ = 50 
SEND_INTERVAL = 1.0 / SEND_HZ
FORCE_DEADBAND = 0.05 

START_MARKER_ID = 20
MARKER_UPDATE_INTERVAL = 5.0
MARKER_SIZE_PX = 1200

def send_cmd(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd + "\n").encode("ascii"))

def drain_telemetry(ser: serial.Serial):
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
            ser.read(2)
    return latest

# =============================================================================
# ── HAUPTPROGRAMM ────────────────────────────────────────────────────────────
# =============================================================================

def main():
    # ── MuJoCo model setup
    model_path = DEFAULT_MODEL
    if not model_path.exists():
        print(f"Warning: {model_path} not found. Trying local 'spiral_chain.xml'")
        model_path = Path("spiral_chain.xml")
        
    spec = mj.MjSpec.from_file(str(model_path))

    cylinder = spec.worldbody.add_body(name="cylinder", pos=[-0.11, 0.00, 0.11])
    cylinder.add_geom(
        name="cyl_geom",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.05, 0.15, 0.05],
        euler=[90, 0, 0],
        rgba=[0.2, 0.8, 0.5, 1],
        density=1000,
    )

    model = spec.compile()
    data = mj.MjData(model)
    print("MuJoCo model loaded.")

    # ── Serial connection
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.001)
        time.sleep(0.1)
        ser.reset_input_buffer()
        print(f"Hardware verbunden: {PORT} @ {BAUDRATE}")
        send_cmd(ser, "start all")
    except Exception as e:
        print(f"Could not open the serial connection (is the hardware on?). Error: {e}")
        return

    # ── ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    current_marker_id = START_MARKER_ID
    cv2.namedWindow("Sync Marker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sync Marker", MARKER_SIZE_PX, MARKER_SIZE_PX)

    records = []
    
    prev_f = [None, None]
    last_send_t = 0.0
    last_marker_t = time.time()

    # ── Phasen-Logik Setup
    current_phase_idx = 0
    phase_start_time = None
    
    # Total duration of all phases, for the user
    total_duration = sum([p.duration for p in PHASES])
    print(f"\nStarte automatische Trajektorien! ({len(PHASES)} Phasen, Total: {total_duration:.1f}s)")

    try:
        with mj.viewer.launch_passive(model, data) as v:
            t0 = time.time()
            phase_start_time = t0
            
            while v.is_running() and current_phase_idx < len(PHASES):
                now = time.time()
                step_start = now
                global_ts = now - t0
                phase_time = now - phase_start_time
                
                # Check for Phase transition
                current_phase = PHASES[current_phase_idx]
                if phase_time >= current_phase.duration:
                    current_phase_idx += 1
                    phase_start_time = now
                    if current_phase_idx >= len(PHASES):
                        print("\nAll phases completed.")
                        break
                    current_phase = PHASES[current_phase_idx]
                    phase_time = 0.0

                # 1) Handle ArUco Marker Window
                if now - last_marker_t >= MARKER_UPDATE_INTERVAL:
                    current_marker_id += 1
                    last_marker_t = now

                marker_img = cv2.aruco.generateImageMarker(aruco_dict, current_marker_id, MARKER_SIZE_PX)
                cv2.imshow("Sync Marker", marker_img)
                cv2.waitKey(1)

                # 2) Calculate Actuator Forces from Profile
                f0 = current_phase.m0.get_force(phase_time, current_phase.duration)
                f1 = current_phase.m1.get_force(phase_time, current_phase.duration)
                
                # Update viewer UI (optional, so you see what the auto-system specifies)
                data.ctrl[0] = -f0
                data.ctrl[1] = -f1

                # 3) Forward to hardware (throttled)
                if now - last_send_t >= SEND_INTERVAL:
                    if prev_f[0] is None or abs(f0 - prev_f[0]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 0 {f0:.2f}")
                        prev_f[0] = f0
                    if prev_f[1] is None or abs(f1 - prev_f[1]) > FORCE_DEADBAND:
                        send_cmd(ser, f"f 1 {f1:.2f}")
                        prev_f[1] = f1
                    last_send_t = now

                # 4) Read Telemetry & Record
                hw = drain_telemetry(ser)
                if hw:
                    hw_ts, hf0, hf1, hr0, hr1 = hw
                    print(f"\r[Phase {current_phase_idx+1}/{len(PHASES)}] time: {global_ts:5.1f}s | "
                          f"target: {f0:5.1f}N {f1:5.1f}N | "
                          f"actual: {hf0:5.1f}N {hf1:5.1f}N (sync:{current_marker_id})   ", end="", flush=True)
                    
                    records.append({
                        "global_timestamp_s": global_ts,
                        "phase_idx": current_phase_idx + 1,
                        "aruco_id": current_marker_id,
                        "cmd_force_0_N": f0,
                        "cmd_force_1_N": f1,
                        "meas_force_0_N": hf0,
                        "meas_force_1_N": hf1,
                        "meas_length_0_mm": hr0,
                        "meas_length_1_mm": hr1,
                        "hw_timestamp_us": hw_ts
                    })

                # 5) Step simulation
                mj.mj_step(model, data)
                v.sync()

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    except KeyboardInterrupt:
        print("\nRecording aborted by the user.")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        send_cmd(ser, "f 0 0.0")
        send_cmd(ser, "f 1 0.0")
        send_cmd(ser, "stop")
        print("\nMotoren gestoppt.")
        ser.close()
        print("Serielle Verbindung geschlossen.")

        # Save data
        if records:
            df = pd.DataFrame(records)
            out_dir = _build_dir("recordings")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "recorded_sys_id_auto_data.csv"
            df.to_csv(out_file, index=False)
            print(f"Data ({len(df)} rows) saved to:\n{out_file}")
        else:
            print("\nKeine Daten aufgezeichnet.")

if __name__ == "__main__":
    main()
