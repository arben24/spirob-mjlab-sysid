"""Interactive preprocessing of a single ring-down recording.

Loads one CSV recording (``spirob_messung_*.csv``), lets you set the start of
the oscillation by hand and apply a low-pass filter, then saves the cleaned
accY signal for evaluation with ``free_vibration.py``.

Controls:
    click on the plot   set the start point directly
    slider "Start"      fine-tune the start point
    slider "Cutoff"     adjust the filter cutoff frequency
    [Filter ON/OFF]     toggle the low-pass filter
    [< Previous]        previous CSV in the same folder
    [Save]              write the cleaned CSV as *_edited.csv
    [Next >]            next CSV in the same folder

Usage::

    uv run sysid/direct/signal_editor.py [path/to/measurement.csv]
    uv run sysid/direct/signal_editor.py       # loads the newest measurement

For most work the GUI in ``free_vibration_gui.py`` is the better tool: it does
the same start-point selection but keeps it as a per-file override in the YAML
instead of writing a second copy of the data.

Inputs/Outputs: data/free_vibration/joint_*/
"""

import csv
import glob
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.signal import butter, filtfilt

from spirob.paths import FREE_VIBRATION_DIR

BUILDS_DIR = str(FREE_VIBRATION_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Dateioperationen
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> dict:
    """
    Load a CSV file and return a dict {sensor_id: {...}}.
    The evaluated signal is accY (no longer the YZ sum).
    """
    with open(filepath) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'accY' not in fieldnames:
            raise ValueError(f"No 'accY' column in {os.path.basename(filepath)}")

        raw: dict = {}
        for row in reader:
            sid = int(row['sensor_id'])
            if sid not in raw:
                raw[sid] = {'t_us': [], 'signal': [], 'accY': [], 'accZ': []}
            raw[sid]['t_us'].append(int(row['t_us']))
            raw[sid]['signal'].append(float(row['accY']))
            raw[sid]['accY'].append(float(row['accY']))
            raw[sid]['accZ'].append(float(row['accZ']))

    result = {}
    for sid, d in raw.items():
        t_us = np.array(d['t_us'], dtype=np.float64)
        result[sid] = {
            't_us': t_us,
            't_s': (t_us - t_us[0]) / 1_000_000.0,
            'signal': np.array(d['signal']),
            'accY':   np.array(d['accY']),
            'accZ':   np.array(d['accZ']),
        }
    return result


def find_csvs_in_dir(directory: str) -> list:
    """All spirob_messung_*.csv in the folder, excluding *_bearbeitet."""
    files = sorted(glob.glob(os.path.join(directory, 'spirob_messung_*.csv')))
    return [f for f in files if '_bearbeitet' not in f]


def find_all_csvs() -> list:
    """All measurement CSVs under the data folder, recursively."""
    files = sorted(glob.glob(os.path.join(BUILDS_DIR, '**', 'spirob_messung_*.csv'),
                             recursive=True))
    return [f for f in files if '_bearbeitet' not in f]


# ──────────────────────────────────────────────────────────────────────────────
# Signalverarbeitung
# ──────────────────────────────────────────────────────────────────────────────

def estimate_fs(t_s: np.ndarray) -> float:
    dt = np.diff(t_s)
    dt_pos = dt[dt > 0]
    return float(1.0 / np.median(dt_pos)) if len(dt_pos) > 0 else 1000.0


def apply_lowpass(sig: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    if cutoff <= 0 or cutoff >= nyq or len(sig) < 3 * order * 2:
        return sig.copy()
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)


# ──────────────────────────────────────────────────────────────────────────────
# Editor-Klasse
# ──────────────────────────────────────────────────────────────────────────────

class SignalEditor:
    def __init__(self, filepaths: list, start_idx: int = 0):
        self.filepaths = filepaths
        self.file_idx  = start_idx
        self.t_start   = 0.0
        self.cutoff    = 50.0
        self.filter_en = True

        self._load_file()
        self._build_ui()

    # ── Load the file ────────────────────────────────────────────────────────

    def _load_file(self):
        fp = self.filepaths[self.file_idx]
        data = load_csv(fp)
        self.sensor_id  = sorted(data.keys())[0]
        sd = data[self.sensor_id]
        self.t_s        = sd['t_s']
        self.signal_raw = sd['signal']
        self.accY_raw   = sd['accY']
        self.accZ_raw   = sd['accZ']
        self.t_us_raw   = sd['t_us']
        self.fs         = estimate_fs(self.t_s)
        self.t_start    = 0.0

    # ── Anzeigedaten ─────────────────────────────────────────────────────────

    def _trimmed(self):
        """Return (t, sig_raw, sig_filt, accY, accZ) from t_start onwards."""
        mask = self.t_s >= self.t_start
        t       = self.t_s[mask]
        sig     = self.signal_raw[mask]
        aY      = self.accY_raw[mask]
        aZ      = self.accZ_raw[mask]
        sig_f   = apply_lowpass(sig, self.fs, self.cutoff) if self.filter_en else sig.copy()
        return t, sig, sig_f, aY, aZ

    # ── UI aufbauen ──────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.close('all')
        self.fig = plt.figure(figsize=(14, 9))

        gs = gridspec.GridSpec(
            5, 4,
            height_ratios=[3.5, 2.2, 0.55, 0.55, 0.65],
            hspace=0.60, wspace=0.40,
            left=0.07, right=0.97, top=0.93, bottom=0.06,
        )

        # ── Oberer Plot: accY (Auswertesignal) ─────────────────────────────
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_main.set_ylabel('accY  (g)')
        self.ax_main.grid(True, ls=':', alpha=0.5)

        self.line_raw,  = self.ax_main.plot([], [], color='#cccccc', lw=0.8,
                                             label='Roh', zorder=1)
        self.line_filt, = self.ax_main.plot([], [], color='#2196F3', lw=1.4,
                                             label='Gefiltert', zorder=2)
        self.vline_main = self.ax_main.axvline(
            x=0, color='#E91E63', lw=1.5, ls='--', label='Startpunkt', zorder=3)
        self.span_main  = self.ax_main.axvspan(0, 0, alpha=0.13, color='#888', zorder=0)
        self.ax_main.legend(loc='upper right', fontsize=9)

        self.info_text = self.ax_main.text(
            0.005, 0.97, '', transform=self.ax_main.transAxes,
            va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85),
        )

        # ── Unterer Plot: Acc Y / Acc Z ───────────────────────────────────
        self.ax_comp = self.fig.add_subplot(gs[1, :], sharex=self.ax_main)
        self.ax_comp.set_xlabel('Time  (s)')
        self.ax_comp.set_ylabel('Acceleration  (g)')
        self.ax_comp.grid(True, ls=':', alpha=0.5)

        self.line_aY, = self.ax_comp.plot([], [], color='#FF9800', lw=0.9, label='Acc Y')
        self.line_aZ, = self.ax_comp.plot([], [], color='#9C27B0', lw=0.9, label='Acc Z')
        self.vline_comp = self.ax_comp.axvline(
            x=0, color='#E91E63', lw=1.5, ls='--', zorder=3)
        self.span_comp  = self.ax_comp.axvspan(0, 0, alpha=0.13, color='#888', zorder=0)
        self.ax_comp.legend(loc='upper right', fontsize=9)

        # ── Slider: Startpunkt ────────────────────────────────────────────
        ax_sl_start = self.fig.add_subplot(gs[2, :])
        t_max = float(self.t_s[-1]) if len(self.t_s) > 1 else 1.0
        self.slider_start = Slider(
            ax_sl_start, 'Start  (s)', 0.0, t_max * 0.95,
            valinit=0.0, color='#E91E63',
        )
        self.slider_start.on_changed(self._on_start)

        # ── Slider: Cutoff ────────────────────────────────────────────────
        ax_sl_cutoff = self.fig.add_subplot(gs[3, :])
        max_cutoff = min(self.fs * 0.45, 500.0)
        self.slider_cutoff = Slider(
            ax_sl_cutoff, 'Cutoff  (Hz)', 1.0, max_cutoff,
            valinit=self.cutoff, color='#4CAF50',
        )
        self.slider_cutoff.on_changed(self._on_cutoff)

        # ── Buttons ───────────────────────────────────────────────────────
        self.btn_prev   = Button(self.fig.add_subplot(gs[4, 0]),
                                 '◀  Vorherige', color='#607D8B', hovercolor='#90A4AE')
        self.btn_filter = Button(self.fig.add_subplot(gs[4, 1]),
                                 'Filter: AN',   color='#4CAF50', hovercolor='#81C784')
        self.btn_save   = Button(self.fig.add_subplot(gs[4, 2]),
                                 'Save',          color='#2196F3', hovercolor='#64B5F6')
        self.btn_next   = Button(self.fig.add_subplot(gs[4, 3]),
                                 'Next  >',   color='#607D8B', hovercolor='#90A4AE')

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_filter.on_clicked(self._on_toggle_filter)
        self.btn_save.on_clicked(self._on_save)
        self.btn_next.on_clicked(self._on_next)

        # A click on the plot sets the start point
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._refresh_title()
        self._update_plot()

    # ── Plot aktualisieren ───────────────────────────────────────────────────

    def _update_plot(self):
        t, sig_raw, sig_filt, aY, aZ = self._trimmed()
        t0 = float(self.t_s[0])
        ts = self.t_start

        # Haupt-Plot
        self.line_raw.set_data(self.t_s, self.signal_raw)
        self.line_filt.set_data(t, sig_filt)
        self.vline_main.set_xdata([ts, ts])
        self._set_span(self.span_main, t0, ts)
        self.ax_main.relim(); self.ax_main.autoscale_view()

        # Komponenten-Plot
        self.line_aY.set_data(self.t_s, self.accY_raw)
        self.line_aZ.set_data(self.t_s, self.accZ_raw)
        self.vline_comp.set_xdata([ts, ts])
        self._set_span(self.span_comp, t0, ts)
        self.ax_comp.relim(); self.ax_comp.autoscale_view()

        # Info-Label
        self.info_text.set_text(
            f'fs ≈ {self.fs:.0f} Hz  │  Start: {ts:.4f} s  │  '
            f'Points: {len(t)}  |  Cutoff: {self.cutoff:.1f} Hz  |  '
            f'Filter: {"ON" if self.filter_en else "OFF"}'
        )

        self.fig.canvas.draw_idle()

    @staticmethod
    def _set_span(span, x0: float, x1: float):
        """Update an axvspan rectangle in place."""
        x1 = max(x1, x0)
        span.set_x(x0)
        span.set_width(x1 - x0)

    def _refresh_title(self):
        fp = self.filepaths[self.file_idx]
        n  = len(self.filepaths)
        title = f'[{self.file_idx + 1}/{n}]  {os.path.basename(fp)}'
        self.fig.canvas.manager.set_window_title(f'Signal-Editor  {title}')
        if hasattr(self, 'ax_main'):
            self.ax_main.set_title(title, fontsize=10)

    # ── Event-Handler ────────────────────────────────────────────────────────

    def _on_start(self, val: float):
        self.t_start = float(val)
        self._update_plot()

    def _on_cutoff(self, val: float):
        self.cutoff = float(val)
        self._update_plot()

    def _on_toggle_filter(self, _):
        self.filter_en = not self.filter_en
        self.btn_filter.label.set_text('Filter: ON' if self.filter_en else 'Filter: OFF')
        self.btn_filter.color = '#4CAF50' if self.filter_en else '#9E9E9E'
        self._update_plot()

    def _on_click(self, event):
        if event.inaxes not in (self.ax_main, self.ax_comp):
            return
        if event.button != 1 or event.xdata is None:
            return
        new_val = float(np.clip(event.xdata,
                                self.slider_start.valmin,
                                self.slider_start.valmax))
        self.slider_start.set_val(new_val)  # triggers _on_start

    def _on_prev(self, _):
        if self.file_idx > 0:
            self.file_idx -= 1
            self._switch_file()

    def _on_next(self, _):
        if self.file_idx < len(self.filepaths) - 1:
            self.file_idx += 1
            self._switch_file()

    def _switch_file(self):
        self._load_file()
        # Adjust the slider bounds
        t_max = float(self.t_s[-1]) if len(self.t_s) > 1 else 1.0
        self.slider_start.valmax = t_max * 0.95
        self.slider_start.ax.set_xlim(0, t_max * 0.95)
        self.slider_start.set_val(0.0)  # triggers _on_start -> _update_plot
        max_cutoff = min(self.fs * 0.45, 500.0)
        self.slider_cutoff.valmax = max_cutoff
        self.slider_cutoff.ax.set_xlim(1.0, max_cutoff)
        self._refresh_title()
        # Reset the status line if it still says "Saved"
        self._refresh_title()

    # ── Saving ───────────────────────────────────────────────────────────────

    def _on_save(self, _):
        mask = self.t_s >= self.t_start
        _, _, sig_filt, _, aZ = self._trimmed()
        t_us_trim = self.t_us_raw[mask]

        fp       = self.filepaths[self.file_idx]
        base     = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(os.path.dirname(fp), f'{base}_bearbeitet.csv')

        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 'accY' = bereinigtes Auswertesignal (getrimmt, ggf. gefiltert);
            # 'accZ' stays raw, for reference.
            writer.writerow(['t_us', 'sensor_id', 'accY', 'accZ'])
            for i in range(len(t_us_trim)):
                writer.writerow([
                    int(t_us_trim[i]),
                    self.sensor_id,
                    f'{sig_filt[i]:.6f}',
                    f'{aZ[i]:.6f}',
                ])

        print(f'Saved: {out_path}  ({len(t_us_trim)} points, '
              f'Start={self.t_start:.4f}s, Cutoff={self.cutoff:.1f}Hz)')
        self.ax_main.set_title(
            f'Saved: {os.path.basename(out_path)}', color='#4CAF50', fontsize=10
        )
        self.fig.canvas.draw_idle()

    # ── Start ────────────────────────────────────────────────────────────────

    def run(self):
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Einstiegspunkt
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f'File not found: {path}')
            sys.exit(1)
        # Load sibling files in the same folder, for navigation
        siblings = find_csvs_in_dir(os.path.dirname(os.path.abspath(path)))
        abs_path = os.path.abspath(path)
        idx = siblings.index(abs_path) if abs_path in siblings else 0
        filepaths = siblings if siblings else [abs_path]
    else:
        filepaths = find_all_csvs()
        if not filepaths:
            print(f'No CSV files found under {BUILDS_DIR}.')
            print('Usage: uv run sysid/direct/signal_editor.py [path.csv]')
            sys.exit(1)
        idx = len(filepaths) - 1  # neueste zuerst
        print(f'{len(filepaths)} measurement(s) found - starting with the newest.')

    SignalEditor(filepaths, start_idx=idx).run()


if __name__ == '__main__':
    main()
