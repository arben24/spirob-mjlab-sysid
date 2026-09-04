"""
frame_parser.py — Gemeinsamer Parser fuer den binaeren ESP32-Sensor-Stream.
============================================================================
EINE Quelle der Wahrheit fuer das Frame-Format. Alle Live-Skripte importieren
von hier, damit eine Formataenderung nicht mehr reihenweise Skripte bricht.

Frame-Format (muss zur Firmware `Custom_PCB/src/main.cpp` passen):
    Header (immer, 12 B): [0xAA,0x55] frame_id(u32) t_esp_us(u32) mode(u8) n(u8)
    pro Sensor je nach mode:
        BOTH (0): sensor_id(u8) + 6 float32 (accX,accY,accZ,magX,magY,magZ)  -> 25 B
        HALL (1): sensor_id(u8) + 3 float32 (magX,magY,magZ)                 -> 13 B
        ACC  (2): sensor_id(u8) + 3 float32 (accX,accY,accZ)                 -> 13 B

`extract_frames()` liefert pro Sensor IMMER eine 6er-Liste
[accX,accY,accZ,magX,magY,magZ]; im jeweiligen Modus fehlende Kanaele sind 0.0.
So funktioniert bei allen Konsumenten weiterhin vals[:3] (Acc) bzw. vals[3:] (Mag).
"""
import struct

# MUSS zu SERIAL_BAUD in der Firmware (Custom_PCB/src/main.cpp) passen!
BAUD_RATE = 1_000_000

FRAME_HEADER = bytes([0xAA, 0x55])
FRAME_HEADER_SIZE = 2
FRAME_META_SIZE = 10  # frame_id(4) + t_esp_us(4) + mode(1) + num_sensors(1)
MAX_SENSORS_PER_FRAME = 16

# Datenmodus im Frame-Header (muss zur Firmware-Enum DataMode passen)
MODE_BOTH = 0
MODE_HALL = 1
MODE_ACC = 2

# Serielles Kommando, um die Firmware in den jeweiligen Modus zu versetzen.
MODE_COMMAND = {MODE_BOTH: b'b', MODE_HALL: b'h', MODE_ACC: b'a'}


class BinaryFrameParser:
    """Parst den binaeren ESP32-Datenstrom mit hoher Robustheit (modus-agnostisch)."""

    def __init__(self, buffer_size: int = 16384, on_text=None):
        """on_text: optionaler Callback(str) fuer ESP32 INFO/WARN-Textzeilen
        (z.B. print). None -> Textzeilen werden still verworfen."""
        self.buffer = bytearray()
        self.buffer_size = buffer_size
        self.on_text = on_text

    def add_data(self, data: bytes):
        self.buffer.extend(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]

    def extract_frames(self):
        frames = []
        while True:
            idx = self.buffer.find(FRAME_HEADER)
            if idx == -1:
                # Kein Header. Nur VOLLSTAENDIGE Zeilen (bis zum letzten \n) an
                # on_text weitergeben -- eine noch nicht fertig angekommene
                # Zeile bliebe sonst abgeschnitten im Log stehen. Ohne \n im
                # Puffer wird nichts verworfen (Puffer waechst weiter, per
                # buffer_size in add_data() ohnehin gedeckelt); ein potenzieller
                # Header-Start am Ende bleibt so in jedem Fall erhalten.
                last_nl = self.buffer.rfind(b'\n')
                if last_nl != -1:
                    discarded = self.buffer[:last_nl + 1]
                    self.buffer = self.buffer[last_nl + 1:]
                    if self.on_text is not None:
                        try:
                            text = discarded.decode('utf-8', errors='ignore')
                            for line in text.splitlines():
                                line = line.strip()
                                if line.startswith("INFO,") or line.startswith("WARN,"):
                                    self.on_text(line)
                        except Exception:
                            pass
                break

            if idx > 0:
                discarded = self.buffer[:idx]
                self.buffer = self.buffer[idx:]
                if self.on_text is not None:
                    try:
                        text = discarded.decode('utf-8', errors='ignore')
                        for line in text.splitlines():
                            line = line.strip()
                            if line.startswith("INFO,") or line.startswith("WARN,"):
                                self.on_text(line)
                    except Exception:
                        pass

            if len(self.buffer) < FRAME_HEADER_SIZE + FRAME_META_SIZE:
                break

            frame_id, t_esp_us, mode, num_sensors = struct.unpack_from("<IIBB", self.buffer, 2)

            # Plausibilitaet: Sensoranzahl UND Modus (faengt Fehl-Header ab)
            if num_sensors == 0 or num_sensors > MAX_SENSORS_PER_FRAME or mode > MODE_ACC:
                self.buffer = self.buffer[2:]
                continue

            n_floats = 6 if mode == MODE_BOTH else 3
            packet_size = 1 + n_floats * 4
            frame_size = FRAME_HEADER_SIZE + FRAME_META_SIZE + num_sensors * packet_size

            if len(self.buffer) < frame_size:
                break

            # Desync-Schutz: echter Frame muss vom naechsten Header gefolgt sein, sonst
            # Fehl-Treffer mitten in den Float-Daten (z.B. nach Byte-Verlust) -> verwerfen.
            if len(self.buffer) < frame_size + FRAME_HEADER_SIZE:
                break
            if self.buffer[frame_size:frame_size + FRAME_HEADER_SIZE] != FRAME_HEADER:
                self.buffer = self.buffer[2:]
                continue

            frame_bytes = self.buffer[:frame_size]
            self.buffer = self.buffer[frame_size:]

            sensors = {}
            offset = FRAME_HEADER_SIZE + FRAME_META_SIZE
            for _ in range(num_sensors):
                sid = frame_bytes[offset]
                vals = struct.unpack(f"<{n_floats}f", frame_bytes[offset + 1:offset + 1 + n_floats * 4])
                # Immer als 6er-Liste [ax,ay,az,mx,my,mz] normieren; fehlende Kanaele = 0.0.
                if mode == MODE_ACC:
                    sensors[sid] = [vals[0], vals[1], vals[2], 0.0, 0.0, 0.0]
                elif mode == MODE_HALL:
                    sensors[sid] = [0.0, 0.0, 0.0, vals[0], vals[1], vals[2]]
                else:  # MODE_BOTH
                    sensors[sid] = list(vals)
                offset += packet_size

            frames.append({"id": frame_id, "mode": mode, "sensors": sensors})
        return frames
