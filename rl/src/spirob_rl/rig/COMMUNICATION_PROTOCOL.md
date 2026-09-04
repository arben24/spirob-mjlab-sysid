# ESP32 ↔ Python Kommunikationsprotokoll (Dual-Motor)

## Übersicht
Die Kommunikation erfolgt bidirektional über UART @ 460800 Baud:
- **ESP32 → Python**: Binäre Sensor-Telemetrie für **2 Motoren** (~600Hz)
- **Python → ESP32**: ASCII-Kommandos mit Motor-Auswahl (zeilenbasiert)

### Motoren
| Index | Servo-ID | Sensor Mux-Channel | Endstop-Pin |
|-------|----------|---------------------|-------------|
| `0`   | 1        | 0                   | 26          |
| `1`   | 2        | 1                   | 27          |

---

## 1. Python → ESP32: ASCII Kommandos

Alle Kommandos sind ASCII-Strings, terminiert mit `\n` (Newline).
Motor-Auswahl: `<m>` = `0`, `1` oder `all`.

### Implementierte Kommandos

| Kommando | Parameter | Beschreibung | Beispiel |
|----------|-----------|--------------|----------|
| `step` | `<m> <speed> <ms>` | Step-Response (force-limited, blocking) | `step 0 500 2000` |
| `v` | `<m\|all> <speed>` | **Direkter Speed-Befehl** (vorzeichenbehaftet) | `v 1 800`, `v all -500` |
| `c` | `<m\|all>` | Homing / Referenzfahrt | `c 0`, `c all` |
| `r` | `<m\|all>` | Return to rope length 0 | `r 1`, `r all` |
| `f` | `<max_force_N>` | Setze globale Kraft-Schwelle (beide Motoren) | `f 15.5` |
| `stop` | `[m]` | Motor stoppen (default: alle) | `stop`, `stop 0` |
| `n` | `<m\|all>` | Nullstelle rope length tracking | `n 0`, `n all` |
| `help` | - | Hilfe anzeigen | `help` |

### Python Beispiel: Seriell verbinden + Kommandos senden

```python
import serial
import time

ser = serial.Serial('/dev/ttyUSB0', baudrate=460800, timeout=0.001)
time.sleep(0.1)
ser.reset_input_buffer()
print("✅ Verbunden mit ESP32 (2 Motoren)")

def send_command(command: str):
    """Sendet ASCII-Kommando an ESP32 (fügt \\n hinzu)."""
    full_cmd = command if command.endswith('\n') else command + '\n'
    ser.write(full_cmd.encode('ascii'))
    print(f"→ Gesendet: {command}")

# Motor 0 Speed setzen
send_command("v 0 800")          # Motor 0 mit Speed 800 vorwärts
time.sleep(2)
send_command("v 0 0")            # Motor 0 stoppen

# Motor 1 Speed setzen
send_command("v 1 -500")         # Motor 1 mit Speed -500 rückwärts
time.sleep(2)

# Beide Motoren gleichzeitig
send_command("v all 600")        # Alle Motoren Speed 600
time.sleep(1)
send_command("stop")             # Alle stoppen

# Force-Limit + Step-Response
send_command("f 12.5")           # Globale Kraft-Schwelle
send_command("step 0 500 2000")  # Motor 0: Step 500 für 2000ms

# Homing
send_command("c all")            # Beide Motoren homen
send_command("c 1")              # Nur Motor 1 homen

ser.close()
```

---

## 2. ESP32 → Python: Binäre Telemetrie (Dual-Motor)

Der ESP32 sendet kontinuierlich (~600Hz) binäre Status-Pakete mit Daten beider Motoren:

### Status-Paket (22 Bytes)

```
┌──────────────┬──────────────┬──────────────────┬──────────────────┬─────────────────────┬─────────────────────┐
│ Header (2B)  │ Timestamp    │ Force Motor 0    │ Force Motor 1    │ Rope Motor 0        │ Rope Motor 1        │
│ 0xAA 0x55    │ uint32 (4B)  │ float32 (4B)     │ float32 (4B)     │ float32 (4B)        │ float32 (4B)        │
└──────────────┴──────────────┴──────────────────┴──────────────────┴─────────────────────┴─────────────────────┘
Gesamt: 2 Header + 20 Payload = 22 Bytes
```

| Feld | Typ | Bytes | Offset | Beschreibung |
|------|-----|-------|--------|--------------|
| Header | `uint8[2]` | 2 | — | Sync-Pattern: `0xAA 0x55` |
| Timestamp | `uint32_t` | 4 | 0 | Mikrosekunden seit MCU-Start |
| Force[0] | `float` | 4 | 4 | Kraft Motor 0 [N] |
| Force[1] | `float` | 4 | 8 | Kraft Motor 1 [N] |
| RopeLen[0] | `float` | 4 | 12 | Seillänge Motor 0 [mm] |
| RopeLen[1] | `float` | 4 | 16 | Seillänge Motor 1 [mm] |

**Endianness**: Little-Endian (Standard für ESP32)

### Step-End Paket (3 Bytes)

```
┌──────────────┬──────────────┐
│ Header (2B)  │ Motor-Index  │
│ 0xBB 0x66    │ uint8 (1B)   │
└──────────────┴──────────────┘
```

| Feld | Typ | Bytes | Beschreibung |
|------|-----|-------|--------------|
| Header | `uint8[2]` | 2 | Step-Ende-Marker: `0xBB 0x66` |
| Motor Index | `uint8_t` | 1 | Welcher Motor fertig ist (0 oder 1) |

### Python Beispiel: Binäre Daten empfangen

```python
import serial
import struct
import time

ser = serial.Serial('/dev/ttyUSB0', baudrate=460800, timeout=0.001)
time.sleep(0.1)
ser.reset_input_buffer()

HEADER = b'\xaa\x55'
STEP_END_HEADER = b'\xbb\x66'
STRUCT_FORMAT = '<I ff ff'       # uint32, float*2, float*2
STRUCT_SIZE = 20                 # 4 + 4*2 + 4*2 = 20 Bytes

def sync_to_header(serial_port) -> bytes | None:
    """Synchronisiert auf Header-Pattern. Gibt Header zurück oder None."""
    sync_buf = b''
    timeout_start = time.time()
    
    while time.time() - timeout_start < 0.1:
        byte = serial_port.read(1)
        if not byte:
            continue
        sync_buf += byte
        if len(sync_buf) > 2:
            sync_buf = sync_buf[-2:]
        
        if sync_buf == HEADER:
            return HEADER
        if sync_buf == STEP_END_HEADER:
            return STEP_END_HEADER
    
    return None

print("📡 Empfange Dual-Motor Daten... (Strg+C zum Beenden)")

try:
    while True:
        hdr = sync_to_header(ser)
        if hdr is None:
            continue
        
        if hdr == HEADER:
            data = ser.read(STRUCT_SIZE)
            if len(data) < STRUCT_SIZE:
                continue
            
            ts_us, f0, f1, r0, r1 = struct.unpack(STRUCT_FORMAT, data)
            print(f"\r{ts_us:10d}us | M0: {f0:6.2f}N {r0:7.1f}mm | M1: {f1:6.2f}N {r1:7.1f}mm", 
                  end='', flush=True)
        
        elif hdr == STEP_END_HEADER:
            motor_byte = ser.read(1)
            motor_idx = motor_byte[0] if motor_byte else -1
            print(f"\n🏁 Step-Ende: Motor {motor_idx}")

except KeyboardInterrupt:
    print("\n👋 Beendet")
finally:
    ser.close()
```

---

## 3. Vollständiges Python-Beispiel (Live-Monitor, Dual-Motor)

```python
#!/usr/bin/env python3
"""
Live-Monitor für ESP32 Dual-Motor Telemetrie.
Zeigt Force + Rope Length beider Motoren in Echtzeit an.
"""
import serial
import struct
import sys
import select
import time

PORT = '/dev/ttyUSB0'
BAUD = 460800
HEADER = b'\xaa\x55'
STEP_END = b'\xbb\x66'
STRUCT_FORMAT = '<I ff ff'
STRUCT_SIZE = 20  # 4 + 8 + 8

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.001)
    time.sleep(0.1)
    ser.reset_input_buffer()
    
    print("✅ Verbunden (2 Motoren) | Befehle: 'v 0 800', 'c all', 'stop', 'q' (quit)")
    
    try:
        while True:
            if ser.in_waiting >= 22:  # Header (2) + Data (20)
                sync_buf = ser.read(2)
                
                if sync_buf == HEADER:
                    data = ser.read(STRUCT_SIZE)
                    if len(data) == STRUCT_SIZE:
                        ts_us, f0, f1, r0, r1 = struct.unpack(STRUCT_FORMAT, data)
                        print(f"\r📡 {ts_us:10d}us | M0: {f0:6.2f}N {r0:7.1f}mm | M1: {f1:6.2f}N {r1:7.1f}mm", 
                              end='', flush=True)
                
                elif sync_buf == STEP_END:
                    m = ser.read(1)
                    print(f"\n🏁 Step-Ende Motor {m[0] if m else '?'}")
            
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                cmd = sys.stdin.readline().strip()
                if cmd == 'q':
                    break
                ser.write((cmd + '\n').encode('ascii'))
                print(f"\n→ {cmd}")
            
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n👋 Beendet")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
```

---

## 4. Wichtige Hinweise

### Timing
- **Status-Pakete**: ~600Hz (beide Motoren in einem Paket)
- **Python Read**: Non-blocking mit `timeout=0.001`
- **Paketgröße**: 22 Bytes (2 Header + 20 Payload)

### Error Handling
```python
try:
    ts_us, f0, f1, r0, r1 = struct.unpack('<I ff ff', data)
except struct.error:
    print("⚠️ Corrupt packet - resync")
    ser.reset_input_buffer()
```

### Motor-Index Konvention
```
Motor 0 → Servo-ID 1, Sensor Mux 0, Endstop Pin 26
Motor 1 → Servo-ID 2, Sensor Mux 1, Endstop Pin 27

CLI benutzt Array-Index (0/1), Servo-Protokoll benutzt IDs (1/2)
```

---

## 5. Zusammenfassung

| Richtung | Protokoll | Baudrate | Beispiel |
|----------|-----------|----------|----------|
| **Python → ESP32** | ASCII (newline-terminated) | 460800 | `b"v 0 800\n"`, `b"step 1 500 2000\n"` |
| **ESP32 → Python** | Binary (packed struct) | 460800 | `0xAA 0x55 + 20 Bytes` |

**Python struct Format:**
```python
STRUCT_FORMAT = '<I ff ff'     # uint32 + 2×float + 2×float = 20 Bytes
ts_us, f0, f1, r0, r1 = struct.unpack(STRUCT_FORMAT, data)
#         ↑   ↑   ↑   ↑
#         M0  M1  M0  M1
#       Force     RopeLen
```
