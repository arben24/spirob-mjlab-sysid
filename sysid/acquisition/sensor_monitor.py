"""Live plot of the accelerometer / magnetometer boards over serial.

Real-time visualisation of several sensor boards via PyQtGraph. Use it to check
that a board is wired up, oriented and sampling correctly *before* recording a
ring-down with ``record_accelerometer.py``.

Binary frame format::

    [0xAA, 0x55] + frame_id(uint32) + t_us(uint32) + num_sensors(uint8)
    per sensor: sensor_id(uint8) + 6 floats (accX, accY, accZ, magX, magY, magZ)

Requires the ``hardware`` and ``gui`` extras:
``uv pip install -e ".[hardware,gui]"``. Adjust SERIAL_PORT below.

Usage::

    uv run sysid/acquisition/sensor_monitor.py
"""

import collections
import queue
import struct
import sys
import threading
import time

import numpy as np
import serial

# PyQtGraph fuer schnelle Visualisierung
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    print("PyQtGraph is not installed. Install it with: uv pip install -e \".[gui]\"")

# ================== EINSTELLUNGEN ==================
SERIAL_PORT = '/dev/ttyUSB0'  # adjust for your system (e.g. COM3 on Windows)
BAUD_RATE = 1000000
MAX_POINTS = 300          # more points give a longer history
UPDATE_INTERVAL = 1    # ~8ms = ca. 120 FP

# Glaettung
SMOOTHING_ALPHA = 0.5  # 0.1 = sehr glatt, 0.5 = schnell

# ================== GLOBALE VARIABLEN ==================
sensor_data = {}
sensor_colors_rgb = {}
color_palette = [
    (255, 100, 100),   # Rot
    (100, 255, 100),   # Gruen
    (100, 100, 255),   # Blau
    (255, 255, 100),   # Gelb
    (255, 100, 255),   # Magenta
    (100, 255, 255),   # Cyan
    (255, 180, 100),   # Orange
    (180, 100, 255),   # Violett
]
color_index = 0

data_queue = queue.Queue(maxsize=1000)
data_lock = threading.Lock()

# ================== BINARY FORMAT ==================
FRAME_HDR_0 = 0xAA
FRAME_HDR_1 = 0x55
FRAME_FIXED_SIZE = 2 + 4 + 4 + 1
SENSOR_PACKET_SIZE = 1 + 6 * 4
MAX_SENSORS_PER_FRAME = 16


def get_sensor_color(sensor_id):
    """Assign a distinct colour to each sensor id."""
    global color_index
    if sensor_id not in sensor_colors_rgb:
        sensor_colors_rgb[sensor_id] = color_palette[color_index % len(color_palette)]
        color_index += 1
    return sensor_colors_rgb[sensor_id]


def create_sensor_entry():
    """Create a new sensor data entry."""
    return {
        'accX': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        'accY': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        'accZ': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        'magX': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        'magY': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        'magZ': collections.deque([0.0]*MAX_POINTS, maxlen=MAX_POINTS),
        # Smoothed values
        'smooth_accX': 0.0,
        'smooth_accY': 0.0,
        'smooth_accZ': 0.0,
        'smooth_magX': 0.0,
        'smooth_magY': 0.0,
        'smooth_magZ': 0.0,
    }


def serial_reader(ser, data_queue, running_event):
    """Thread body: fast read of the serial data (binary frames)."""
    buffer = bytearray()
    while running_event.is_set():
        try:
            if ser.in_waiting:
                buffer.extend(ser.read(ser.in_waiting))
            else:
                time.sleep(0.001)

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

                # Sanity check to avoid false headers
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
                    try:
                        data_queue.put_nowait((sensor_id, accX, accY, accZ, magX, magY, magZ))
                    except queue.Full:
                        try:
                            data_queue.get_nowait()
                            data_queue.put_nowait((sensor_id, accX, accY, accZ, magX, magY, magZ))
                        except Exception:
                            pass
                    offset += SENSOR_PACKET_SIZE
        except Exception as e:
            print(f"Lesefehler: {e}")
            break


def process_data(line):
    """Process one data row or binary tuple, with smoothing."""
    global sensor_data

    try:
        if isinstance(line, tuple):
            sensor_id, raw_accX, raw_accY, raw_accZ, raw_magX, raw_magY, raw_magZ = line
        else:
            parts = line.split(',')
            if len(parts) != 8:
                return False
            sensor_id = int(parts[1])
            raw_accX = float(parts[2])
            raw_accY = float(parts[3])
            raw_accZ = float(parts[4])
            raw_magX = float(parts[5])
            raw_magY = float(parts[6])
            raw_magZ = float(parts[7])
        
        with data_lock:
            if sensor_id not in sensor_data:
                sensor_data[sensor_id] = create_sensor_entry()
                get_sensor_color(sensor_id)
                print(f"Neuer Sensor erkannt: ID {sensor_id}")
            
            d = sensor_data[sensor_id]
            alpha = SMOOTHING_ALPHA
            
            # EMA Glaettung
            d['smooth_accX'] = alpha * raw_accX + (1-alpha) * d['smooth_accX']
            d['smooth_accY'] = alpha * raw_accY + (1-alpha) * d['smooth_accY']
            d['smooth_accZ'] = alpha * raw_accZ + (1-alpha) * d['smooth_accZ']
            d['smooth_magX'] = alpha * raw_magX + (1-alpha) * d['smooth_magX']
            d['smooth_magY'] = alpha * raw_magY + (1-alpha) * d['smooth_magY']
            d['smooth_magZ'] = alpha * raw_magZ + (1-alpha) * d['smooth_magZ']
            
            # Smoothed values speichern
            d['accX'].append(d['smooth_accX'])
            d['accY'].append(d['smooth_accY'])
            d['accZ'].append(d['smooth_accZ'])
            d['magX'].append(d['smooth_magX'])
            d['magY'].append(d['smooth_magY'])
            d['magZ'].append(d['smooth_magZ'])
        
        return True
    except ValueError:
        return False


class FastSensorVisualizer:
    """Fast visualisation with PyQtGraph."""
    
    def __init__(self):
        # PyQtGraph Konfiguration
        pg.setConfigOptions(antialias=True, useOpenGL=True)
        
        # Application and window
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title="Sensor Visualisierung - FAST")
        self.win.resize(1400, 800)
        self.win.show()
        
        # Plots erstellen - 2 Zeilen, 3 Spalten
        # Row 1: acceleration
        self.plot_accX = self.win.addPlot(title="Acceleration X (g)")
        self.plot_accY = self.win.addPlot(title="Acceleration Y (g)")
        self.plot_accZ = self.win.addPlot(title="Acceleration Z (g)")
        
        self.win.nextRow()
        
        # Row 2: magnetic field
        self.plot_magX = self.win.addPlot(title="Magnetfeld X (uT)")
        self.plot_magY = self.win.addPlot(title="Magnetfeld Y (uT)")
        self.plot_magZ = self.win.addPlot(title="Magnetfeld Z (uT)")
        
        # Achsen konfigurieren
        for plot in [self.plot_accX, self.plot_accY, self.plot_accZ]:
            plot.setYRange(-1.5, 1.5)
            plot.setXRange(0, MAX_POINTS)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.addLegend()
        
        for plot in [self.plot_magX, self.plot_magY, self.plot_magZ]:
            plot.setYRange(-500, 500)
            plot.setXRange(0, MAX_POINTS)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.addLegend()
        
        # Kurven-Dictionary
        self.curves = {}
        
        # Rate counter
        self.last_time = time.time()
        self.packet_count = 0
        self.hz = 0.0
        
        # Serial
        self.ser = None
        self.running_event = threading.Event()
        self.reader_thread = None
        
        # Timer fuer Updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        
        # X-Achse
        self.x_data = np.arange(MAX_POINTS)
    
    def _create_curves_for_sensor(self, sensor_id):
        """Create the curves for a new sensor."""
        color = sensor_colors_rgb[sensor_id]
        pen = pg.mkPen(color=color, width=2)
        name = f"ID {sensor_id}"
        
        self.curves[sensor_id] = {
            'accX': self.plot_accX.plot(pen=pen, name=name),
            'accY': self.plot_accY.plot(pen=pen, name=name),
            'accZ': self.plot_accZ.plot(pen=pen, name=name),
            'magX': self.plot_magX.plot(pen=pen, name=name),
            'magY': self.plot_magY.plot(pen=pen, name=name),
            'magZ': self.plot_magZ.plot(pen=pen, name=name),
        }
    
    def update(self):
        """Update callback, driven by the timer."""
        # Drain the queue
        processed = 0
        while not data_queue.empty() and processed < 300:  # raised for higher data rates
            try:
                line = data_queue.get_nowait()
                process_data(line)
                processed += 1
                self.packet_count += 1
            except queue.Empty:
                break
        
        # Compute the rate, once per second
        now = time.time()
        dt = now - self.last_time
        if dt >= 1.0:
            num_sensors = len(sensor_data)
            if num_sensors > 0:
                # Divide by the sensor count: one packet per sensor per time step.
                self.hz = (self.packet_count / num_sensors) / dt
            else:
                self.hz = 0.0
            self.packet_count = 0
            self.last_time = now
            self.win.setWindowTitle(f"Sensor Visualisierung - FAST | Frequenz: {self.hz:.1f} Hz ({num_sensors} Sensoren)")
            print(f"Frequenz: {self.hz:.1f} Hz ({num_sensors} aktive Sensoren)")
        
        # Kurven aktualisieren
        with data_lock:
            for sensor_id, data in sensor_data.items():
                if sensor_id not in self.curves:
                    self._create_curves_for_sensor(sensor_id)
                
                # Data as numpy arrays
                self.curves[sensor_id]['accX'].setData(self.x_data, np.array(data['accX']))
                self.curves[sensor_id]['accY'].setData(self.x_data, np.array(data['accY']))
                self.curves[sensor_id]['accZ'].setData(self.x_data, np.array(data['accZ']))
                self.curves[sensor_id]['magX'].setData(self.x_data, np.array(data['magX']))
                self.curves[sensor_id]['magY'].setData(self.x_data, np.array(data['magY']))
                self.curves[sensor_id]['magZ'].setData(self.x_data, np.array(data['magZ']))
    
    def start(self):
        """Start the visualisation."""
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print(f"Port {SERIAL_PORT} geoeffnet @ {BAUD_RATE} baud")
            # Flush the serial buffer and put the ESP32 into BOTH mode
            self.ser.reset_input_buffer()
            self.ser.write(b'b')
            print("ESP32 switched to BOTH mode, buffer cleared.")
        except Exception as e:
            print(f"Port error: {e}")
            print("Starte Demo-Modus...")
            self._start_demo_mode()
            return
        
        # Reader Thread starten
        self.running_event.set()
        self.reader_thread = threading.Thread(
            target=serial_reader,
            args=(self.ser, data_queue, self.running_event)
        )
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        # Timer starten
        self.timer.start(UPDATE_INTERVAL)
        
        # App starten (blockiert bis Fenster geschlossen)
        self.app.exec()
        
        # Aufraeumen
        self._cleanup()
    
    def _start_demo_mode(self):
        """Demo mode with simulated data."""
        import time
        
        def demo_generator():
            t = 0
            while self.running_event.is_set():
                for sensor_id in [0, 32]:
                    noise = np.random.normal(0, 0.1)
                    accX = np.sin(t * 0.1 + sensor_id * 0.5) * 1.5 + noise
                    accY = np.cos(t * 0.15 + sensor_id * 0.3) * 1.2 + noise
                    accZ = np.sin(t * 0.2 + sensor_id * 0.7) + 1 + noise
                    
                    magX = np.sin(t * 0.05 + sensor_id) * 50 + noise * 10
                    magY = np.cos(t * 0.08 + sensor_id * 0.4) * 40 + noise * 10
                    magZ = np.sin(t * 0.03 + sensor_id * 0.2) * 30 + 20 + noise * 10
                    
                    line = (sensor_id, accX, accY, accZ, magX, magY, magZ)
                    try:
                        data_queue.put_nowait(line)
                    except Exception:
                        pass
                
                t += 1
                time.sleep(0.01)
        
        self.running_event.set()
        demo_thread = threading.Thread(target=demo_generator)
        demo_thread.daemon = True
        demo_thread.start()
        
        self.timer.start(UPDATE_INTERVAL)
        self.app.exec()
        self._cleanup()
    
    def _cleanup(self):
        """Aufraeumen."""
        self.running_event.clear()
        self.timer.stop()
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Port geschlossen.")


# ================== MATPLOTLIB FALLBACK ==================
class MatplotlibVisualizer:
    """Fallback when PyQtGraph is unavailable."""
    
    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.plt = plt
        self.FuncAnimation = FuncAnimation
        
        self.fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        self.fig.suptitle('Sensor Visualisierung (Matplotlib - langsam)')
        
        self.ax_accX, self.ax_accY, self.ax_accZ = axes[0]
        self.ax_magX, self.ax_magY, self.ax_magZ = axes[1]
        
        for ax in [self.ax_accX, self.ax_accY, self.ax_accZ]:
            ax.set_ylim(-4, 4)
            ax.set_xlim(0, MAX_POINTS)
            ax.grid(True, alpha=0.3)
        
        for ax in [self.ax_magX, self.ax_magY, self.ax_magZ]:
            ax.set_ylim(-100, 100)
            ax.set_xlim(0, MAX_POINTS)
            ax.grid(True, alpha=0.3)
        
        self.ax_accX.set_title('Acc X (g)')
        self.ax_accY.set_title('Acc Y (g)')
        self.ax_accZ.set_title('Acc Z (g)')
        self.ax_magX.set_title('Mag X (uT)')
        self.ax_magY.set_title('Mag Y (uT)')
        self.ax_magZ.set_title('Mag Z (uT)')
        
        self.lines = {}
        self.ser = None
        self.running_event = threading.Event()
        self.reader_thread = None
    
    def _create_lines(self, sensor_id):
        color = [c/255 for c in sensor_colors_rgb[sensor_id]]
        label = f'ID {sensor_id}'
        self.lines[sensor_id] = {
            'accX': self.ax_accX.plot([], [], color=color, label=label, linewidth=2)[0],
            'accY': self.ax_accY.plot([], [], color=color, linewidth=2)[0],
            'accZ': self.ax_accZ.plot([], [], color=color, linewidth=2)[0],
            'magX': self.ax_magX.plot([], [], color=color, linewidth=2)[0],
            'magY': self.ax_magY.plot([], [], color=color, linewidth=2)[0],
            'magZ': self.ax_magZ.plot([], [], color=color, linewidth=2)[0],
        }
        self.ax_accX.legend(loc='upper right', fontsize=8)
    
    def update(self, frame):
        while not data_queue.empty():
            try:
                line = data_queue.get_nowait()
                process_data(line)
            except queue.Empty:
                break
        
        with data_lock:
            x_range = range(MAX_POINTS)
            for sensor_id, data in sensor_data.items():
                if sensor_id not in self.lines:
                    self._create_lines(sensor_id)
                
                self.lines[sensor_id]['accX'].set_data(x_range, list(data['accX']))
                self.lines[sensor_id]['accY'].set_data(x_range, list(data['accY']))
                self.lines[sensor_id]['accZ'].set_data(x_range, list(data['accZ']))
                self.lines[sensor_id]['magX'].set_data(x_range, list(data['magX']))
                self.lines[sensor_id]['magY'].set_data(x_range, list(data['magY']))
                self.lines[sensor_id]['magZ'].set_data(x_range, list(data['magZ']))
        
        return []
    
    def start(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"Port {SERIAL_PORT} geoeffnet")
            # Flush the serial buffer and put the ESP32 into BOTH mode
            self.ser.reset_input_buffer()
            self.ser.write(b'a')
            print("ESP32 switched to BOTH mode, buffer cleared.")
        except Exception as e:
            print(f"Port error: {e}")
            self._start_demo()
            return
        
        self.running_event.set()
        self.reader_thread = threading.Thread(
            target=serial_reader,
            args=(self.ser, data_queue, self.running_event)
        )
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        self.ani = self.FuncAnimation(
            self.fig, self.update,
            interval=50,  # 20 FPS max fuer matplotlib
            blit=False
        )
        
        self.plt.tight_layout()
        self.plt.show()
        self._cleanup()
    
    def _start_demo(self):
        import time
        
        def demo():
            t = 0
            while self.running_event.is_set():
                for sid in [0, 32]:
                        line = (sid,
                            float(np.sin(t*0.1)*1.5),
                            float(np.cos(t*0.1)*1.2),
                            1.0,
                            float(np.sin(t*0.05)*50),
                            float(np.cos(t*0.05)*40),
                            30.0)
                        data_queue.put(line)
                t += 1
                time.sleep(0.02)
        
        self.running_event.set()
        threading.Thread(target=demo, daemon=True).start()
        
        self.ani = self.FuncAnimation(self.fig, self.update, interval=50, blit=False)
        self.plt.tight_layout()
        self.plt.show()
        self._cleanup()
    
    def _cleanup(self):
        self.running_event.clear()
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()


def main():
    print("=" * 60)
    print("  Sensor Daten Visualisierung - OPTIMIERT")
    print("=" * 60)
    print(f"  Port: {SERIAL_PORT} @ {BAUD_RATE} baud")
    print(f"  Glaettung: alpha = {SMOOTHING_ALPHA}")
    print(f"  Update: {UPDATE_INTERVAL}ms (~{1000/UPDATE_INTERVAL:.0f} FPS)")
    print("=" * 60)
    
    if HAS_PYQTGRAPH:
        print("  Verwende PyQtGraph (SCHNELL)")
        visualizer = FastSensorVisualizer()
    else:
        print("  Verwende Matplotlib (langsam)")
        print("  Fuer bessere Performance: pip install pyqtgraph PyQt6")
        visualizer = MatplotlibVisualizer()
    
    print("=" * 60)
    print()
    
    visualizer.start()


if __name__ == "__main__":
    main()
