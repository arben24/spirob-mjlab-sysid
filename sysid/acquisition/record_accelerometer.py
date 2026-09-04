"""Record an accelerometer ring-down from the ESP32 sensor board.

Reads the binary telemetry stream, waits for a trigger (a jump in acceleration
larger than TRIGGER_THRESHOLD_G, i.e. the moment the segment is released), and
writes a window around it -- PRE_RECORD_TIME_S before, RECORD_TIME_S after --
to a CSV that ``free_vibration.py`` and its GUI can read.

The hardware timestamp ``t_us`` is used rather than the host clock, so the
samples are exactly evenly spaced. The serial reader deliberately contains no
``time.sleep`` so it can keep up with the full sample rate.

Requires the ``hardware`` extra and a connected board:
``uv pip install -e ".[hardware]"``. Adjust SERIAL_PORT below.

Usage::

    uv run sysid/acquisition/record_accelerometer.py

Outputs: build/recordings/spirob_messung_<timestamp>.csv (+ a preview PNG).
Move the CSVs into data/free_vibration/joint_NN/ to evaluate them.
"""

import collections
import csv
import math
import os
import struct
from datetime import datetime

import matplotlib.pyplot as plt
import serial

from spirob.paths import build_dir as _build_dir

# ================== EINSTELLUNGEN ==================
SERIAL_PORT = '/dev/ttyUSB0'  # Anpassen!
BAUD_RATE = 1000000

# Recording settings
TRIGGER_THRESHOLD_G = 0.5     # change in 'g' between two samples that fires the trigger
PRE_RECORD_TIME_S = 0.5       # seconds kept from BEFORE the trigger
RECORD_TIME_S = 1.0           # seconds recorded AFTER the trigger
MAX_SAMPLE_RATE_HZ = 1000     # max sensor sample rate; sets the buffer size

# Sensor filter: None accepts every sensor, an int processes only that sensor id
TARGET_SENSOR_ID = None

# ================== BINARY FORMAT ==================
FRAME_HDR_0 = 0xAA
FRAME_HDR_1 = 0x55
FRAME_FIXED_SIZE = 2 + 4 + 4 + 1
SENSOR_PACKET_SIZE = 1 + 6 * 4
MAX_SENSORS_PER_FRAME = 16

def ensure_builds_dir():
    builds_dir = str(_build_dir('recordings'))
    os.makedirs(builds_dir, exist_ok=True)
    return builds_dir

def parse_serial_stream(ser, callback):
    """High-rate read of the binary stream, with no artificial delays."""
    buffer = bytearray()
    
    # Put the ESP32 into the right mode if needed and flush the buffer
    ser.reset_input_buffer()
    ser.write(b'b')
    
    while True:
        try:
            # Blocking read: max(1, in_waiting) reads immediately when data is
            # there, without spinning the CPU at 100 % when it is not.
            bytes_to_read = max(1, ser.in_waiting)
            new_data = ser.read(bytes_to_read)
            if new_data:
                buffer.extend(new_data)

            while True:
                idx = buffer.find(bytes([FRAME_HDR_0, FRAME_HDR_1]))
                if idx == -1:
                    if len(buffer) > 1:
                        buffer = buffer[-1:]
                    break

                if idx > 0:
                    del buffer[:idx]

                if len(buffer) < FRAME_FIXED_SIZE:
                    break

                frame_id, t_us, n = struct.unpack_from("<IIB", buffer, 2)

                if n == 0 or n > MAX_SENSORS_PER_FRAME:
                    del buffer[:2]
                    continue

                total_len = FRAME_FIXED_SIZE + n * SENSOR_PACKET_SIZE
                if len(buffer) < total_len:
                    break

                payload = buffer[FRAME_FIXED_SIZE:total_len]
                del buffer[:total_len]

                offset = 0
                for _ in range(n):
                    pkt = payload[offset:offset + SENSOR_PACKET_SIZE]
                    sensor_id = pkt[0]
                    accX, accY, accZ, magX, magY, magZ = struct.unpack("<ffffff", pkt[1:])
                    
                    callback(t_us, frame_id, sensor_id, accX, accY, accZ, magX, magY, magZ)
                    offset += SENSOR_PACKET_SIZE
                    
        except KeyboardInterrupt:
            print("\nManuell abgebrochen.")
            break
        except Exception as e:
            print(f"Lesefehler: {e}")
            break

def main():
    builds_dir = ensure_builds_dir()
    
    max_history = int(PRE_RECORD_TIME_S * MAX_SAMPLE_RATE_HZ * 2)
    pre_trigger_buffer = collections.deque(maxlen=max_history)
    
    recorded_data = []
    
    is_recording = False
    trigger_t_us = 0      # Speichert den *Hardware*-Zeitstempel des Triggers
    last_acc = {}

    print("=" * 50)
    print(" Waiting for the serial connection ...")
    print("=" * 50)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"Connected. Waiting for motion (threshold {TRIGGER_THRESHOLD_G} g) ...")
    except Exception as e:
        print(f"Error opening the port: {e}")
        return

    def on_data_received(t_us, frame_id, sensor_id, accX, accY, accZ, magX, magY, magZ):
        nonlocal is_recording, trigger_t_us, last_acc, pre_trigger_buffer, recorded_data

        if TARGET_SENSOR_ID is not None and sensor_id != TARGET_SENSOR_ID:
            return

        # YZ-Summe berechnen
        acc_sum_YZ = accY + accZ

        row = (t_us, frame_id, sensor_id, accX, accY, accZ, acc_sum_YZ, magX, magY, magZ)

        if not is_recording:
            pre_trigger_buffer.append(row)

            # Schwellwert-Erkennung: euklidischer Abstand in YZ-Ebene.
            # X stays constant as the gravity axis; the motion appears in Y and Z.
            if sensor_id in last_acc:
                l_y, l_z = last_acc[sensor_id]
                delta = math.sqrt((accY - l_y)**2 + (accZ - l_z)**2)
                
                if delta >= TRIGGER_THRESHOLD_G:
                    print(f"\n[!] TRIGGER fired on sensor {sensor_id}. Delta = {delta:.2f} g")
                    is_recording = True
                    trigger_t_us = t_us  # Hardware-Zeitpunkt des Triggers speichern!
                    
                    recorded_data.extend(pre_trigger_buffer)
                    print(f"Recording for {RECORD_TIME_S} seconds ...")
            
            last_acc[sensor_id] = (accY, accZ)
            
        else:
            recorded_data.append(row)
            
            # Stop after RECORD_TIME_S, measured on the hardware clock.
            if (t_us - trigger_t_us) >= (RECORD_TIME_S * 1_000_000):
                raise StopIteration("recording complete")

    try:
        parse_serial_stream(ser, on_data_received)
    except StopIteration:
        pass
    finally:
        ser.close()
        print("Serielle Verbindung geschlossen.")

    if not recorded_data:
        print("No data recorded.")
        return

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(builds_dir, f"spirob_messung_{timestamp_str}.csv")
    plot_filename = os.path.join(builds_dir, f"spirob_plot_{timestamp_str}.png")

    print(f"\nSpeichere {len(recorded_data)} Datenpunkte...")

    # CSV Schreiben
    headers = ['t_us', 'frame_id', 'sensor_id', 'accX', 'accY', 'accZ', 'acc_sum_YZ', 'magX', 'magY', 'magZ']
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(recorded_data)

    # Plot erstellen
    sensors = {}
    for row in recorded_data:
        t_us, fid, sid, ax, ay, az, asum_yz, mx, my, mz = row
        if sid not in sensors:
            sensors[sid] = {'t': [], 'ay': [], 'az': [], 'asum_yz': []}

        # Time axis: the trigger sits at 0.0 s
        t_relative_s = (t_us - trigger_t_us) / 1_000_000.0

        if t_relative_s >= -PRE_RECORD_TIME_S:
            sensors[sid]['t'].append(t_relative_s)
            sensors[sid]['ay'].append(ay)
            sensors[sid]['az'].append(az)
            sensors[sid]['asum_yz'].append(asum_yz)

    num_sensors = len(sensors)
    fig, axes = plt.subplots(num_sensors, 1, figsize=(10, 3.5 * num_sensors), sharex=True)
    if num_sensors == 1:
        axes = [axes]
    
    fig.suptitle("SpiRob joint ring-down (YZ sum)", fontsize=14)

    for idx, (sid, sdata) in enumerate(sensors.items()):
        ax = axes[idx]
        
        ax.plot(sdata['t'], sdata['ay'], label='Acc Y', linewidth=1.0, alpha=0.5)
        ax.plot(sdata['t'], sdata['az'], label='Acc Z', linewidth=1.0, alpha=0.5)
        ax.plot(sdata['t'], sdata['asum_yz'], label='Summe YZ', linewidth=1.5, color='black')
        
        # Mark the trigger instant (exactly at 0)
        ax.axvline(x=0.0, color='r', linestyle='--', label='Trigger')
        
        ax.set_title(f"Sensor ID: {sid}")
        ax.set_ylabel("Acceleration (g)")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time after trigger (s)")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    print(f" -> CSV and plot written to {builds_dir}")

if __name__ == '__main__':
    main()