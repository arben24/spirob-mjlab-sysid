# Data acquisition

**Everything here needs the physical setup.** These scripts are included so the
recordings in `data/` are reproducible and so the hardware protocol is
documented — not because they run on a clean checkout.

```bash
uv pip install -e ".[hardware,vision,gui]"
```

## TL;DR

| Script | Hardware | What it does |
|---|---|---|
| `record_trajectory.py` | ESP32 motor controller + camera | drives both tendons through scripted force phases, shows an ArUco sync marker, logs commanded/measured force and rope length to CSV |
| `track_and_sync.py` | video file (no live hardware) | detects joint markers 0–13 and sync markers ≥20, aligns video to CSV, merges into one Parquet |
| `trim_dataset.py` | none | cuts samples off both ends and resets time and frame counters to 0 |
| `overlay_video.py` | video file | renders joint angles and forces back onto the frames, for visual verification |
| `record_accelerometer.py` | ESP32 sensor board | triggered ring-down capture for the free-vibration test |
| `sensor_monitor.py` | ESP32 sensor board | live plot, to check a board before recording |
| `digital_twin_bridge.py` | ESP32 motor controller | drag the MuJoCo actuator sliders, forces go to the real motors |

## The recording chain

```
record_trajectory.py  ──▶  CSV (forces, rope lengths, timestamps)
        +                                    │
   GoPro video (ArUco markers)  ─────────────┤
                                             ▼
                                    track_and_sync.py
                                             │
                                             ▼
                                    trim_dataset.py
                                             │
                                             ▼
                          data/trajectories/*.parquet  ──▶  real2sim.py
```

## Serial protocol

Both boards stream binary frames at high rate.

**Motor controller** (460800 baud):
`[0xAA 0x55] + uint32 ts_us + float force[2] + float rope_mm[2] + [0xBB 0x66]`

**Sensor board** (1000000 baud):
`[0xAA 0x55] + uint32 frame_id + uint32 t_us + uint8 num_sensors`, then per
sensor `uint8 sensor_id + 6 floats (accX, accY, accZ, magX, magY, magZ)`.

The **hardware timestamp** `t_us` is authoritative, not the host clock — that is
what makes the samples evenly spaced enough for the logarithmic-decrement
evaluation.

Set `SERIAL_PORT` / `PORT` at the top of each script (default `/dev/ttyUSB0`).

## ArUco setup

* Markers **0–13** are glued to the segments; the angle between successive
  markers is the joint angle.
* Markers **≥20** are shown on screen by `record_trajectory.py` and step every
  5 s. They are what aligns the video timeline to the CSV — without them there
  is no common clock between camera and motor controller.
* Dictionary: `DICT_4X4_50`.

## Output convention

`track_and_sync.py` produces exactly what `real2sim.py` consumes:

| Column | Meaning |
|---|---|
| `joint_1_deg` … `joint_13_deg` | joint angles, **joint 1 = base** |
| `meas_force_0_N`, `meas_force_1_N` | measured tendon forces (the simulation's input) |
| `cmd_force_0_N`, `cmd_force_1_N` | commanded forces |
| `meas_length_0_mm`, `meas_length_1_mm` | rope lengths |
| `global_timestamp_s` | common time base |

Recordings land in `build/recordings/`. Move a finished one into
`data/trajectories/` to make it the reference.
