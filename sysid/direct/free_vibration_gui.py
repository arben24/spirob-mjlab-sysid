"""Interactive GUI for the free-vibration (ring-down) system identification.

Every detection and filter parameter can be tuned live; the signal trace, the
detected peaks, the envelope and the current result (w_d, zeta, k, d) update
immediately. This is where the *authoritative* per-joint numbers come from --
in particular the moment of inertia J, which is hand-tuned here and which k and
d scale linearly with.

Features:
  * navigate every measurement of a folder (left / right)
  * live evaluation of the current measurement
  * averaging over the whole folder, with outlier marking
  * save all settings + results to <folder>/sysid_settings.yaml
  * on start, existing settings are read back from that YAML

Usage::

    uv run sysid/direct/free_vibration_gui.py [folder]
    # without an argument: the first joint folder under data/free_vibration/

Figure export::

    uv run sysid/direct/free_vibration_gui.py [folder] --figure
    # renders the interface windowless to build/free_vibration/gui/ --
    # gui_<folder>_full plus one crop per section (signal, results,
    # parameters, navigation), each as PDF and PNG. The state matches the
    # stored sysid_settings.yaml: what is in the YAML is in the figure.

Appearance:
  Colours, font sizes and surfaces come from ``spirob.plotstyle`` -- the same
  tokens as every other figure. GUI_FONT_SCALE raises all fonts, because the
  interface is scaled down to text width in the document. Numbers deliberately
  keep the decimal point here: they show the state of the software and the
  YAML, not typeset chart values.

Inputs/Outputs: data/free_vibration/joint_*/ (reads and writes the YAML)
"""

import datetime
import glob
import os
import sys

import matplotlib
import numpy as np
import yaml

# --figure renders the interface windowless; the backend must be chosen
# before the first pyplot import.
if '--figure' in sys.argv or '--plot' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

# free_vibration.py sits next to this file -> importable on direct start
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import free_vibration as tss  # noqa: E402

from spirob import plotstyle as ts  # noqa: E402

BUILDS_DIR = tss.BUILDS_DIR
SETTINGS_FILENAME = 'sysid_settings.yaml'

# ── Abbildungsgeometrie ───────────────────────────────────────────────────────
# The interface *is* a Matplotlib figure, so the export is not a
# screenshot but a vector PDF of the very same figure.
FIG_W, FIG_H = 15.5, 9.0

# The interface is scaled down to text width in the document (factor ~0.4).
# Without enlargement 9 pt text ends up below 4 pt. The factor raises all
# fonts together; crops of individual sections are less
# heavily scaled down and therefore look a little more generously set.
GUI_FONT_SCALE = 1.45

# Regions in figure coordinates (x0, y0, x1, y1) -- they also define the
# backdrop of the control panels and the crops used for the export.
REGIONS = {
    'signal':     dict(box=(0.022, 0.492, 0.688, 0.982), title='Signal trace',
                       panel=False),
    'results':     dict(box=(0.698, 0.118, 0.990, 0.982), title='Results',
                       panel=True),
    'parameter':  dict(box=(0.022, 0.118, 0.688, 0.480), title='Analysis parameters',
                       panel=True),
    'navigation': dict(box=(0.022, 0.008, 0.990, 0.112), title='Navigation and saving',
                       panel=True),
}

# Uniform appearance of the sliders and check boxes
SLIDER_TRACK = dict(track_color=ts.UI_FACE,
                    handle_style=dict(facecolor='white', edgecolor=ts.INK_2,
                                      size=13))


def _f(x, n: int) -> float:
    """Round to a native float -- YAML cannot serialise numpy scalars."""
    return round(float(x), n)

# Defaults, used when no YAML exists yet
DEFAULT_SETTINGS = {
    'sensor_id': 0,
    'J': 0.001,
    'lowpass_enabled': True,
    'lowpass_cutoff_hz': 50.0,
    'lowpass_order': 4,
    'peak_prominence_factor': 0.1,
    'peak_min_distance_s': 0.02,
    'min_peaks_required': 3,
    'outlier_sigma': 2.0,
    'trigger_mode': 'onset',
}


# ──────────────────────────────────────────────────────────────────────────────
# Folders / files
# ──────────────────────────────────────────────────────────────────────────────

def list_recording_folders() -> list:
    """Sub-folders of data/free_vibration/ that contain measurements."""
    folders = []
    for entry in sorted(os.listdir(BUILDS_DIR)):
        full = os.path.join(BUILDS_DIR, entry)
        if not os.path.isdir(full) or entry == 'results':
            continue
        if glob.glob(os.path.join(full, 'spirob_messung_*.csv')):
            folders.append(full)
    return folders


def list_recordings(folder: str) -> list:
    """
    Every measurement of a folder. If a recording has a
    signal_editor produced a '*_bearbeitet.csv', that one is preferred.
    """
    raw = sorted(glob.glob(os.path.join(folder, 'spirob_messung_*.csv')))
    raw = [f for f in raw if '_bearbeitet' not in f]

    chosen = []
    for f in raw:
        base = os.path.splitext(f)[0]
        edited = f'{base}_bearbeitet.csv'
        chosen.append(edited if os.path.isfile(edited) else f)
    return chosen


def settings_path(folder: str) -> str:
    return os.path.join(folder, SETTINGS_FILENAME)


def load_settings(folder: str) -> tuple:
    """
    Load settings from the YAML (defaults for missing keys).

    Returns:
        (settings: dict, start_overrides_by_name: dict[str, float])
    """
    settings = dict(DEFAULT_SETTINGS)
    overrides = {}
    path = settings_path(folder)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            loaded = data.get('settings', data)  # akzeptiere beide Layouts
            for key in DEFAULT_SETTINGS:
                if key in loaded and loaded[key] is not None:
                    settings[key] = loaded[key]
            overrides = loaded.get('start_overrides', {}) or {}
            print(f"Settings loaded from: {path}")
        except Exception as e:
            print(f"  Could not read {path} ({e}) - using the defaults.")
    else:
        print("No sysid_settings.yaml found - using the defaults.")
    return settings, overrides


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class SysIdGUI:
    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.folder_name = os.path.basename(self.folder.rstrip('/'))
        self.settings, overrides_by_name = load_settings(self.folder)

        # Load and cache every measurement once (t_s, signal)
        self.recordings = []
        for path in list_recordings(self.folder):
            try:
                t_s, signal = tss.load_csv(path, int(self.settings['sensor_id']))
                self.recordings.append({'path': path, 't_s': t_s, 'signal': signal})
            except Exception as e:
                print(f"  {os.path.basename(path)} skipped: {e}")

        if not self.recordings:
            raise RuntimeError(
                f"No usable measurements in {self.folder} "
                f"(sensor_id={self.settings['sensor_id']})."
            )

        # Manual start-point overrides (file path -> time in s), read from the YAML
        name2path = {os.path.basename(r['path']): r['path'] for r in self.recordings}
        self.start_overrides = {}
        for name, t in overrides_by_name.items():
            if name in name2path and t is not None:
                self.start_overrides[name2path[name]] = float(t)

        self.idx = 0
        self._loading = False          # suppresses recompute during a bulk set
        self._delete_armed = False     # two-click guard for deletion
        self._last_results = []        # list[tss.MeasurementResult]
        self._last_summary = None
        self._n_failed = 0

        self._build_ui()
        self._apply_settings_to_widgets()
        self._recompute()

    # ── Parameters from the widgets ──────────────────────────────────────────

    @property
    def params(self) -> 'tss.AnalysisParams':
        return tss.AnalysisParams(
            lowpass_enabled=self.chk_filter.get_status()[0],
            lowpass_cutoff_hz=float(self.s_cutoff.val),
            lowpass_order=int(round(self.s_order.val)),
            peak_prominence_factor=float(self.s_prom.val),
            peak_min_distance_s=float(self.s_dist.val) / 1000.0,  # ms → s
            min_peaks_required=int(round(self.s_minpk.val)),
            outlier_sigma=float(self.s_sigma.val),
            trigger_mode='onset' if self.chk_onset.get_status()[0] else 'jump',
        )

    @property
    def J(self) -> float:
        try:
            return float(self.tb_J.text)
        except ValueError:
            return float(self.settings['J'])

    @property
    def sensor_id(self) -> int:
        try:
            return int(float(self.tb_sensor.text))
        except ValueError:
            return int(self.settings['sensor_id'])

    def current_settings_dict(self) -> dict:
        p = self.params
        return {
            'sensor_id': int(self.sensor_id),
            'J': _f(self.J, 8),
            'lowpass_enabled': bool(p.lowpass_enabled),
            'lowpass_cutoff_hz': _f(p.lowpass_cutoff_hz, 4),
            'lowpass_order': int(p.lowpass_order),
            'peak_prominence_factor': _f(p.peak_prominence_factor, 4),
            'peak_min_distance_s': _f(p.peak_min_distance_s, 5),
            'min_peaks_required': int(p.min_peaks_required),
            'outlier_sigma': _f(p.outlier_sigma, 3),
            'trigger_mode': p.trigger_mode,
            'start_overrides': {
                os.path.basename(pth): _f(t, 5)
                for pth, t in sorted(self.start_overrides.items())
            },
        }

    # ── Trigger / Startpunkt ──────────────────────────────────────────────────

    def _auto_trigger_idx(self, t_s, signal, params) -> int:
        """Automatically detected start index, per the selected mode."""
        if params.trigger_mode == 'onset':
            return tss.find_onset_index(t_s, signal, params.onset_noise_factor)
        return tss.find_trigger_index(t_s, signal)

    def _trigger_idx_for(self, rec, params):
        """Start index of a measurement: manual override, or None (= automatic)."""
        ov = self.start_overrides.get(rec['path'])
        if ov is None:
            return None
        t_s = rec['t_s']
        return int(np.clip(np.searchsorted(t_s, ov), 0, len(t_s) - 1))

    def _effective_trigger_idx(self, rec, params) -> int:
        """The start index actually used (override, else auto) -- for display."""
        ov_idx = self._trigger_idx_for(rec, params)
        if ov_idx is not None:
            return ov_idx
        return self._auto_trigger_idx(rec['t_s'], rec['signal'], params)

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _compute_file(self, t_s, signal, params, J, trigger_idx=None):
        """Analyse one measurement. Returns (fs, analysis|None, result|None)."""
        fs = tss.estimate_sample_rate(t_s)
        analysis = tss.analyze_free_vibration(t_s, signal, fs, params,
                                              trigger_idx=trigger_idx)
        if analysis is None:
            return fs, None, None
        p = tss.compute_parameters(analysis['omega_d'], analysis['zeta'], J)
        result = dict(analysis)
        result.update(p)
        result['fs'] = fs
        return fs, analysis, result

    def _compute_summary(self, params, J):
        """Evaluate every measurement -> (results, summary, n_failed)."""
        results = []
        n_failed = 0
        for rec in self.recordings:
            tidx = self._trigger_idx_for(rec, params)
            _, _, res = self._compute_file(rec['t_s'], rec['signal'], params, J, tidx)
            if res is None:
                n_failed += 1
                continue
            results.append(tss.MeasurementResult(
                filename=rec['path'], joint_name=self.folder_name,
                sensor_id=self.sensor_id, J=J,
                omega_d=res['omega_d'], omega_n=res['omega_n'],
                f_d=res['f_d'], f_n=res['f_n'],
                zeta=res['zeta'], delta=res['delta'],
                k=res['k'], d=res['d'], n_peaks=res['n_peaks'],
            ))
        results = tss.mark_outliers(results, params.outlier_sigma)
        summary = tss.compute_joint_summary(self.folder_name, results)
        return results, summary, n_failed

    # ── UI aufbauen ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        plt.close('all')
        ts.apply_style(scale=GUI_FONT_SCALE)

        self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        self.fig.canvas.manager.set_window_title(
            f'Torsionsfeder Sys-ID – {self.folder_name}')

        self._draw_regions()

        # ── Plots ────────────────────────────────────────────────────────────
        # Only one axis now: the old bar panel of peak amplitudes merely
        # repeated what the envelope already shows.
        self.ax_sig = self.fig.add_axes([0.062, 0.556, 0.608, 0.362])
        self.ax_sig.set_xlabel('Time (s)')
        self.ax_sig.set_ylabel('accY (zentriert)')
        ts.grid_on(self.ax_sig)

        # ── Info-Panel (Text) ────────────────────────────────────────────────
        self.ax_info = self.fig.add_axes([0.714, 0.372, 0.268, 0.545])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.0, 1.0, '', transform=self.ax_info.transAxes,
            va='top', ha='left', family='monospace',
            fontsize=ts.plt.rcParams['font.size'] * 0.92,
            color=ts.INK, linespacing=1.45,
        )

        # ── Start-point slider (per measurement) + auto button ───────────────
        self.s_start = Slider(self.fig.add_axes([0.158, 0.415, 0.295, 0.021]),
                              'Start point (s)  ', 0.0, 1.0, valinit=0.0,
                              color=ts.C2, valfmt='%.3f', **SLIDER_TRACK)
        self.s_start.on_changed(self._on_start_change)
        self.btn_auto = self._button([0.505, 0.408, 0.080, 0.036], 'Auto')
        self.btn_auto.on_clicked(self._on_auto_start)

        # ── Check boxes: onset detection & low pass ──────────────────────────
        ax_onset = self.fig.add_axes([0.048, 0.332, 0.24, 0.040])
        ax_onset.set_facecolor('none')
        ax_onset.set_navigate(False)
        for sp in ax_onset.spines.values():
            sp.set_visible(False)
        self.chk_onset = CheckButtons(ax_onset, ['Onset detection'],
                                      [self.settings['trigger_mode'] == 'onset'],
                                      **self._check_props())
        self.chk_onset.on_clicked(self._on_change)

        ax_chk = self.fig.add_axes([0.048, 0.280, 0.24, 0.040])
        ax_chk.set_facecolor('none')
        ax_chk.set_navigate(False)
        for sp in ax_chk.spines.values():
            sp.set_visible(False)
        self.chk_filter = CheckButtons(ax_chk, ['Low pass on'], [True],
                                       **self._check_props())
        self.chk_filter.on_clicked(self._on_change)

        # ── Text boxes: J and sensor_id ──────────────────────────────────────
        self.tb_J = TextBox(self.fig.add_axes([0.158, 0.216, 0.095, 0.032]),
                            'J (kg m^2)  ', initial=str(self.settings['J']),
                            color=ts.SURFACE, hovercolor=ts.UI_PANEL)
        self.tb_J.on_submit(self._on_change)
        self.tb_sensor = TextBox(self.fig.add_axes([0.158, 0.162, 0.095, 0.032]),
                                 'sensor_id  ', initial=str(self.settings['sensor_id']),
                                 color=ts.SURFACE, hovercolor=ts.UI_PANEL)
        self.tb_sensor.on_submit(self._on_sensor_change)

        # ── Sliders (global parameters) ──────────────────────────────────────
        # Own column on the right so the labels do not reach into the control
        # group on the left.
        sx, sw, sh = 0.437, 0.175, 0.021
        rows = (0.352, 0.310, 0.268, 0.226, 0.184, 0.142)
        self.s_cutoff = Slider(self.fig.add_axes([sx, rows[0], sw, sh]),
                               'Cutoff (Hz)  ', 1.0, 480.0, valinit=50.0,
                               valfmt='%.0f', color=ts.C1, **SLIDER_TRACK)
        self.s_order  = Slider(self.fig.add_axes([sx, rows[1], sw, sh]),
                               'Order  ', 1, 8, valinit=4, valstep=1,
                               valfmt='%d', color=ts.C1, **SLIDER_TRACK)
        self.s_prom   = Slider(self.fig.add_axes([sx, rows[2], sw, sh]),
                               'Prominence  ', 0.01, 0.6, valinit=0.1,
                               valfmt='%.2f', color=ts.C1, **SLIDER_TRACK)
        self.s_dist   = Slider(self.fig.add_axes([sx, rows[3], sw, sh]),
                               'Peak distance (ms)  ', 1, 80, valinit=20, valstep=1,
                               valfmt='%d', color=ts.C1, **SLIDER_TRACK)
        self.s_minpk  = Slider(self.fig.add_axes([sx, rows[4], sw, sh]),
                               'Min. peaks  ', 2, 15, valinit=3, valstep=1,
                               valfmt='%d', color=ts.C1, **SLIDER_TRACK)
        self.s_sigma  = Slider(self.fig.add_axes([sx, rows[5], sw, sh]),
                               'Outlier sigma  ', 0.5, 4.0, valinit=2.0,
                               valfmt='%.1f', color=ts.C1, **SLIDER_TRACK)
        for s in (self.s_start, self.s_cutoff, self.s_order, self.s_prom,
                  self.s_dist, self.s_minpk, self.s_sigma):
            if s is not self.s_start:
                s.on_changed(self._on_change)
            s.vline.set_visible(False)       # the init marker is distracting in print
            s.label.set_color(ts.INK)
            s.valtext.set_color(ts.INK_2)

        # ── Buttons ──────────────────────────────────────────────────────────
        self.btn_prev = self._button([0.048, 0.020, 0.104, 0.044], '< Previous')
        self.btn_next = self._button([0.162, 0.020, 0.104, 0.044], 'Next >')
        self.btn_delete = self._button([0.292, 0.020, 0.152, 0.044],
                                       'Delete recording', kind='danger')
        self.btn_reset = self._button([0.456, 0.020, 0.128, 0.044], 'Defaults')
        self.btn_plot = self._button([0.596, 0.020, 0.178, 0.044],
                                     'Export plot', kind='accent')
        self.btn_save = self._button([0.786, 0.020, 0.188, 0.044],
                                     'Save (YAML)', kind='accent')
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_delete.on_clicked(self._on_delete)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_plot.on_clicked(self._on_export_plot)
        self.btn_save.on_clicked(self._on_save)

    # ── Interface chrome ─────────────────────────────────────────────────────

    def _check_props(self) -> dict:
        """Check boxes large and high-contrast -- scaled down for print, a
        standard tick mark is no longer recognisable as one."""
        return dict(
            frame_props=dict(s=280, facecolor='white', edgecolor=ts.INK_2,
                             linewidth=1.2),
            check_props=dict(s=280, facecolor=ts.C1, linewidths=2.4),
            label_props=dict(color=[ts.INK],
                             fontsize=[ts.plt.rcParams['font.size']]),
        )

    def _button(self, rect, label, kind='neutral'):
        """Button with a uniform appearance.

        ``kind``: 'neutral' (default), 'accent' (primary action),
        'danger' (destructive action). The meaning is always also in the
        label -- colour reinforces, it is never the only cue.
        """
        face, hover, fg = {
            'neutral': (ts.UI_FACE, ts.UI_FACE_HOVER, ts.INK),
            'accent': (ts.UI_ACCENT, ts.UI_ACCENT_HOVER, 'white'),
            'danger': (ts.UI_DANGER, ts.UI_DANGER_HOVER, ts.INK),
        }[kind]
        ax = self.fig.add_axes(rect)
        ax.set_navigate(False)
        btn = Button(ax, label, color=face, hovercolor=hover)
        btn.label.set_color(fg)
        btn.label.set_fontsize(ts.plt.rcParams['font.size'])
        for spine in ax.spines.values():
            spine.set_edgecolor(ts.UI_PANEL_EDGE)
            spine.set_linewidth(0.8)
        return btn

    def _draw_regions(self):
        """Draw the control-panel backdrops and label each section.

        The groups double as the crops that :meth:`export_figures` saves
        individually -- taking a single section into the text still gives
        together with its heading.
        """
        head_size = ts.plt.rcParams['font.size'] * 1.12
        for spec in REGIONS.values():
            x0, y0, x1, y1 = spec['box']
            if spec['panel']:
                self.fig.add_artist(FancyBboxPatch(
                    (x0, y0), x1 - x0, y1 - y0,
                    boxstyle='round,pad=0,rounding_size=0.008',
                    transform=self.fig.transFigure,
                    facecolor=ts.UI_PANEL, edgecolor=ts.UI_PANEL_EDGE,
                    linewidth=0.9, zorder=0))
            self.fig.text(x0 + 0.012, y1 - 0.022, spec['title'],
                          fontsize=head_size, color=ts.INK, weight='bold',
                          va='center', ha='left', zorder=2)

    def _apply_settings_to_widgets(self):
        """Push self.settings into the widgets (without intermediate recompute)."""
        self._loading = True
        s = self.settings
        self.s_cutoff.set_val(float(s['lowpass_cutoff_hz']))
        self.s_order.set_val(int(s['lowpass_order']))
        self.s_prom.set_val(float(s['peak_prominence_factor']))
        self.s_dist.set_val(float(s['peak_min_distance_s']) * 1000.0)
        self.s_minpk.set_val(int(s['min_peaks_required']))
        self.s_sigma.set_val(float(s['outlier_sigma']))
        # Bring the check buttons into the desired state
        if self.chk_filter.get_status()[0] != bool(s['lowpass_enabled']):
            self.chk_filter.set_active(0)
        if self.chk_onset.get_status()[0] != (s['trigger_mode'] == 'onset'):
            self.chk_onset.set_active(0)
        self.tb_J.set_val(str(s['J']))
        self.tb_sensor.set_val(str(s['sensor_id']))
        self._loading = False

    # ── Recompute + Plot ────────────────────────────────────────────────────

    def _recompute(self):
        if self._loading or not self.recordings:
            return
        params = self.params
        J = self.J

        rec = self.recordings[self.idx]
        # Move the start-point slider to the current value (override or auto)
        self._sync_start_slider(rec, params)

        tidx = self._trigger_idx_for(rec, params)
        fs, analysis, result = self._compute_file(rec['t_s'], rec['signal'], params, J, tidx)

        self._draw_signal(rec, analysis, result, params)

        results, summary, n_failed = self._compute_summary(params, J)
        self._last_results = results
        self._last_summary = summary
        self._n_failed = n_failed

        self._update_info(rec, fs, result, summary, n_failed)
        self.fig.canvas.draw_idle()

    def _sync_start_slider(self, rec, params):
        """Fit the start-point slider to the current measurement (without triggering an override)."""
        self._loading = True
        t_s = rec['t_s']
        t_max = float(t_s[-1]) if len(t_s) > 1 else 1.0
        self.s_start.valmax = t_max
        self.s_start.ax.set_xlim(0, t_max)
        ov = self.start_overrides.get(rec['path'])
        if ov is None:
            auto_idx = self._auto_trigger_idx(t_s, rec['signal'], params)
            val = float(t_s[auto_idx])
        else:
            val = float(ov)
        self.s_start.set_val(min(val, t_max))
        self._loading = False

    def _render_signal(self, ax, rec, analysis, result, params,
                       print_mode: bool = False):
        """Draw the signal trace into ``ax``.

        The same rendering serves two targets: the on-screen interface and the
        standalone figure for the document. ``print_mode`` only raises line
        widths, marker sizes and labels -- the content
        is identical, so the figure shows exactly what the GUI showed.
        """
        # Line widths: in print the figure is not scaled down, but it is
        # rasterised or greyscaled -- thin lines break away.
        if print_mode:
            lw_raw, lw_filt, lw_env, lw_start = 1.0, 2.2, 1.8, 2.0
            ms, raw_color = 8.0, '#bdbcb3'
        else:
            lw_raw, lw_filt, lw_env, lw_start = 0.9, 1.5, 1.5, 1.6
            ms, raw_color = 7.0, ts.FAINT

        t_s, signal = rec['t_s'], rec['signal']

        # Determine the start index consistently with the analysis
        is_manual = self.start_overrides.get(rec['path']) is not None
        trig = self._effective_trigger_idx(rec, params)

        # Base rendering, even when no peaks were found
        fs = tss.estimate_sample_rate(t_s)
        pre = signal[:max(trig, 1)]
        offset = float(np.mean(pre)) if len(pre) > 5 else 0.0
        sig_post = (signal - offset)[trig:]
        t_abs = t_s[trig:]
        if params.lowpass_enabled:
            filt = tss.apply_lowpass_filter(sig_post, fs,
                                            params.lowpass_cutoff_hz, params.lowpass_order)
        else:
            filt = sig_post.copy()

        # A manually set start is additionally marked by its line style,
        # not by colour alone.
        if is_manual:
            start_label = 'Start (manual)'
            # Long dash rather than solid: in greyscale print the legend key
            # would otherwise be indistinguishable from the signal trace.
            start_kw = dict(color=ts.C2, linestyle=(0, (6, 2.5)), linewidth=lw_start)
        else:
            start_label = f'Start ({params.trigger_mode})'
            start_kw = dict(color=ts.INK_2, linestyle=(0, (1.2, 1.6)),
                            linewidth=lw_start)

        ax.axhline(0, color=ts.BASELINE, lw=0.9, zorder=1)
        ax.plot(t_s, signal - offset, color=raw_color, lw=lw_raw,
                label='Raw (centred)', zorder=2)
        ax.plot(t_abs, filt, label='Filtered', zorder=3, **ts.line_kw(0, width=lw_filt))
        ax.axvline(t_s[trig], label=start_label, zorder=4, **start_kw)

        if result is not None:
            pos = analysis['pos_peak_idx']
            neg = analysis['neg_peak_idx']
            allp = analysis['all_peak_idx']
            ax.plot(t_abs[pos], filt[pos], label='Maxima', zorder=6,
                    **ts.marker_kw(1, size=ms, marker='v'))
            ax.plot(t_abs[neg], filt[neg], label='Minima', zorder=6,
                    **ts.marker_kw(2, size=ms, marker='^'))

            # Upper envelope through the peak maxima (maxima = reference, lie below).
            # Decay rate from a log-linear fit of the peaks; the amplitude anchored so
            # so the curve touches the highest maximum and all peaks lie below it.
            peak_t = analysis['t_post'][allp]
            peak_a = np.abs(filt[allp])
            fit_mask = peak_a > 1e-9
            if fit_mask.sum() >= 2:
                slope, _ = np.polyfit(peak_t[fit_mask], np.log(peak_a[fit_mask]), 1)
                decay = -slope  # Abklingrate [1/s]
                if decay <= 0:  # the fit does not decay (e.g. run-up) -> use the physical rate
                    decay = result['zeta'] * result['omega_n']
                # C = upper bound: env(t_i) = C*exp(-decay*t_i) >= every maximum
                C = float(np.max(peak_a[fit_mask] * np.exp(decay * peak_t[fit_mask])))
                env = C * np.exp(-decay * analysis['t_post'])
                rate = ts.de_num(decay, 1) if print_mode else f'{decay:.1f}'
                ax.plot(t_abs, env, label=f'Envelope (decay {rate}/s)',
                        **ts.model_kw(width=lw_env))
                ax.plot(t_abs, -env, **ts.model_kw(width=lw_env))

            # Put a generous time window around the ring-down: the start
            # should be visible with some lead-in, and the decay down to rest.
            t0 = float(t_s[trig])
            t_last = float(t_abs[allp][-1])
            w = max(t_last - t0, 1e-3)
            lo = max(float(t_s[0]), t0 - 0.60 * w)
            hi = min(float(t_s[-1]), t_last + 1.00 * w)
            ax.set_xlim(lo, hi)

            vis = (t_s >= lo) & (t_s <= hi)
            if vis.any():
                yspan = float(np.nanmax(np.abs((signal - offset)[vis])))
                ax.set_ylim(-yspan * 1.16, yspan * 1.16)
        else:
            ax.text(0.5, 0.5, 'Not enough peaks - adjust the parameters',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=ts.plt.rcParams['font.size'] * 1.1, color=ts.WARN,
                    bbox=dict(boxstyle='round,pad=0.5', fc=ts.WARN_BG, ec=ts.WARN,
                              linewidth=1.0))

        # Legend in a fixed reading order: curves first, then markers.
        # Without it, the two-column layout mixes related entries.
        order = ('Raw', 'Filtered', 'Envelope', 'Start', 'Maxima', 'Minima')
        handles, labels = ax.get_legend_handles_labels()
        rank = {lab: next((i for i, k in enumerate(order) if lab.startswith(k)),
                          len(order)) for lab in labels}
        pairs = sorted(zip(handles, labels), key=lambda hl: rank[hl[1]])
        ax.legend([h for h, _ in pairs], [lab for _, lab in pairs],
                  loc='upper right', ncol=2, columnspacing=1.2,
                  framealpha=0.92, facecolor=ts.SURFACE, edgecolor='none',
                  frameon=True)

    def _draw_signal(self, rec, analysis, result, params):
        ax = self.ax_sig
        ax.clear()
        ts.grid_on(ax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('accY (centred)')
        self._render_signal(ax, rec, analysis, result, params)
        ax.set_title(
            f'[{self.idx + 1}/{len(self.recordings)}]  '
            f'{os.path.basename(rec["path"])}', color=ts.INK_2)

    def _update_info(self, rec, fs, result, summary, n_failed):
        lines = []
        lines.append('CURRENT MEASUREMENT')
        lines.append('─' * 40)
        lines.append(f'[{self.idx + 1}/{len(self.recordings)}] {os.path.basename(rec["path"])}')
        lines.append(f'fs ≈ {fs:.0f} Hz')
        lines.append('')
        if result is not None:
            lines.append(f"ω_d = {result['omega_d']:8.2f} rad/s   (f_d {result['f_d']:.2f} Hz)")
            lines.append(f"ω_n = {result['omega_n']:8.2f} rad/s   (f_n {result['f_n']:.2f} Hz)")
            lines.append(f"ζ   = {result['zeta']:8.4f}")
            lines.append(f"δ   = {result['delta']:8.4f}")
            lines.append(f"k   = {result['k']:10.5f} Nm/rad")
            lines.append(f"d   = {result['d']:10.6f} Nm·s/rad")
            lines.append(f"Peaks: {result['n_peaks']}")
        else:
            lines.append('No valid evaluation')
            lines.append('  (too few peaks)')

        lines.append('')
        lines.append('AVERAGE OVER ALL MEASUREMENTS')
        lines.append('─' * 40)
        if summary is not None and summary.n_valid > 0:
            lines.append(f'measurements: {summary.n_measurements}   '
                         f'valid: {summary.n_valid}   '
                         f'outliers: {summary.n_outliers}')
            if n_failed:
                lines.append(f'without peaks: {n_failed}')
            lines.append('')
            lines.append(f'ω_d = {summary.omega_d_mean:8.2f} ± {summary.omega_d_std:.2f} rad/s')
            lines.append(f'ω_n = {summary.omega_n_mean:8.2f} ± {summary.omega_n_std:.2f} rad/s')
            lines.append(f'ζ   = {summary.zeta_mean:8.4f} ± {summary.zeta_std:.4f}')
            lines.append(f'k   = {summary.k_mean:10.5f} ± {summary.k_std:.5f} Nm/rad')
            lines.append(f'd   = {summary.d_mean:10.6f} ± {summary.d_std:.6f} Nm·s/rad')
        else:
            lines.append('No valid measurement')
            if n_failed:
                lines.append(f'without peaks: {n_failed}')

        self.info_text.set_text('\n'.join(lines))

    # ── Event-Handler ────────────────────────────────────────────────────────

    def _on_change(self, _=None):
        self._recompute()

    def _on_start_change(self, val):
        """Start point moved manually -> set an override for the current measurement."""
        if self._loading:
            return
        rec = self.recordings[self.idx]
        self.start_overrides[rec['path']] = float(val)
        self._recompute()

    def _on_auto_start(self, _):
        """Discard the manual start point of the current measurement (back to auto)."""
        rec = self.recordings[self.idx]
        self.start_overrides.pop(rec['path'], None)
        self._recompute()

    def _on_sensor_change(self, _=None):
        """sensor_id changed -> reload the CSVs with the new sensor_id."""
        if self._loading:
            return
        new_sid = self.sensor_id
        reloaded = []
        for rec in self.recordings:
            try:
                t_s, signal = tss.load_csv(rec['path'], new_sid)
                reloaded.append({'path': rec['path'], 't_s': t_s, 'signal': signal})
            except Exception:
                pass  # file has no such sensor_id -> skip it
        if reloaded:
            self.recordings = reloaded
            self.idx = min(self.idx, len(self.recordings) - 1)
        else:
            print(f'  No data for sensor_id={new_sid} - keeping the previous selection.')
        self._recompute()

    def _on_prev(self, _):
        self._disarm_delete()
        if self.idx > 0:
            self.idx -= 1
            self._recompute()

    def _on_next(self, _):
        self._disarm_delete()
        if self.idx < len(self.recordings) - 1:
            self.idx += 1
            self._recompute()

    # ── Delete recording (two-click guard) ───────────────────────────────────

    def _recording_files(self, path: str) -> list:
        """Every file belonging to one recording (raw, edited, plot PNG)."""
        if path.endswith('_bearbeitet.csv'):
            raw = path[:-len('_bearbeitet.csv')] + '.csv'
        else:
            raw = path
        base = os.path.splitext(raw)[0]                 # …/spirob_messung_<ts>
        edited = base + '_bearbeitet.csv'
        plot = base.replace('spirob_messung_', 'spirob_plot_') + '.png'
        return [f for f in (raw, edited, plot) if os.path.isfile(f)]

    def _disarm_delete(self):
        if self._delete_armed:
            self._delete_armed = False
            self.btn_delete.label.set_text('Delete recording')
            self.btn_delete.ax.set_facecolor('#FF8A65')
            self.fig.canvas.draw_idle()

    def _on_delete(self, _):
        # First click: only arm the button
        if not self._delete_armed:
            self._delete_armed = True
            self.btn_delete.label.set_text('Really delete?')
            self.btn_delete.ax.set_facecolor('#E53935')
            self.fig.canvas.draw_idle()
            return

        # Second click: actually delete
        self._delete_armed = False
        self.btn_delete.label.set_text('Delete recording')
        self.btn_delete.ax.set_facecolor('#FF8A65')

        rec = self.recordings[self.idx]
        for f in self._recording_files(rec['path']):
            try:
                os.remove(f)
                print(f'Deleted: {f}')
            except OSError as e:
                print(f'  Could not delete {f}: {e}')

        self.start_overrides.pop(rec['path'], None)
        del self.recordings[self.idx]

        if not self.recordings:
            print('No measurements left in the folder - closing the GUI.')
            plt.close(self.fig)
            return

        self.idx = min(self.idx, len(self.recordings) - 1)
        self._recompute()

    def _on_export_plot(self, _):
        """Save the current signal trace as a standalone figure."""
        try:
            written = self.export_signal_plot()
        except Exception as e:
            print(f'⚠ Plot-Export fehlgeschlagen: {e}')
            self.ax_sig.set_title(f'⚠ Export fehlgeschlagen: {e}', color=ts.WARN)
        else:
            name = os.path.basename(written[0])
            self.ax_sig.set_title(f'✓ Plot exportiert: {name}', color=ts.GOOD)
        self.fig.canvas.draw_idle()

    def _on_reset(self, _):
        self.settings = dict(DEFAULT_SETTINGS)
        self._apply_settings_to_widgets()
        self._recompute()

    def _on_save(self, _):
        path = settings_path(self.folder)
        settings = self.current_settings_dict()

        results_block = None
        s = self._last_summary
        if s is not None:
            per_file = []
            for r in self._last_results:
                per_file.append({
                    'file': os.path.basename(r.filename),
                    'omega_d': _f(r.omega_d, 4),
                    'omega_n': _f(r.omega_n, 4),
                    'f_n': _f(r.f_n, 4),
                    'zeta': _f(r.zeta, 6),
                    'delta': _f(r.delta, 6),
                    'k': _f(r.k, 8),
                    'd': _f(r.d, 8),
                    'n_peaks': int(r.n_peaks),
                    'outlier': bool(r.is_outlier),
                })
            results_block = {
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'folder': self.folder_name,
                'n_measurements': int(s.n_measurements),
                'n_valid': int(s.n_valid),
                'n_outliers': int(s.n_outliers),
                'n_failed': int(self._n_failed),
                'omega_d_mean': _f(s.omega_d_mean, 4),
                'omega_d_std': _f(s.omega_d_std, 4),
                'omega_n_mean': _f(s.omega_n_mean, 4),
                'omega_n_std': _f(s.omega_n_std, 4),
                'zeta_mean': _f(s.zeta_mean, 6),
                'zeta_std': _f(s.zeta_std, 6),
                'k_mean': _f(s.k_mean, 8),
                'k_std': _f(s.k_std, 8),
                'd_mean': _f(s.d_mean, 8),
                'd_std': _f(s.d_std, 8),
                'per_file': per_file,
            }

        doc = {'settings': settings}
        if results_block is not None:
            doc['results'] = results_block

        with open(path, 'w') as f:
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        print(f'Saved: {path}')
        self.ax_sig.set_title(f'Saved: {SETTINGS_FILENAME}',
                              color=ts.GOOD)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()

    # ── Export for the document ──────────────────────────────────────────────

    def export_signal_plot(self, out_dir: str = None, stem: str = None) -> list:
        """Only the signal trace -- as a standalone figure, not as a crop.

        A crop of the interface comes from a 15.5 inch wide
        figure, scaled down to text width in the document, which makes
        everything small. This figure is laid out at text width from the start
        (``ts.FIG_FULL``) and embedded without scaling -- type at
        document size, lines at print weight.

        The rcParams are set inside an ``rc_context`` so the
        enlargement of the interface stays untouched.
        """
        rec = self.recordings[self.idx]
        params, J = self.params, self.J
        tidx = self._trigger_idx_for(rec, params)
        _, analysis, result = self._compute_file(rec['t_s'], rec['signal'],
                                                 params, J, tidx)

        out_dir = out_dir or os.path.join(tss.RESULTS_DIR, 'gui')
        if stem is None:
            base = os.path.splitext(os.path.basename(rec['path']))[0]
            stem = f'plot_{self.folder_name}_{base}'

        with plt.rc_context():
            ts.apply_style()                       # document size, no uplift
            fig, ax = plt.subplots(figsize=(ts.FIG_FULL, 3.5))
            ts.grid_on(ax)
            ax.set_xlabel('Time $t$ (s)')
            ax.set_ylabel('Acceleration accY, centred (g)')
            self._render_signal(ax, rec, analysis, result, params, print_mode=True)
            ts.german_axes(fig)
            fig.tight_layout()
            written = ts.save(fig, os.path.join(out_dir, stem))
        return written

    def export_figures(self, out_dir: str, prefix: str = 'gui') -> list:
        """Save the interface as a figure -- the whole thing and each section.

        Not a screenshot: the interface *is* a Matplotlib figure, so the
        export is a vector PDF (plus a PNG for viewing). Besides the full
        view, each section from :data:`REGIONS` is saved
        cropped, so the text can show just one part of it.
        """
        os.makedirs(out_dir, exist_ok=True)
        self.fig.canvas.draw()
        written = list(self.export_signal_plot(out_dir, f'{prefix}_plot'))

        stem = os.path.join(out_dir, f'{prefix}_full')
        for ext in ('pdf', 'png'):
            path = f'{stem}.{ext}'
            self.fig.savefig(path, dpi=300, facecolor=ts.SURFACE)
            written.append(path)
            print(f'  → {path}')

        pad = 0.006   # etwas Luft um den Ausschnitt, in Figurkoordinaten
        for name, spec in REGIONS.items():
            x0, y0, x1, y1 = spec['box']
            bbox = Bbox.from_extents((x0 - pad) * FIG_W, (y0 - pad) * FIG_H,
                                     (x1 + pad) * FIG_W, (y1 + pad) * FIG_H)
            stem = os.path.join(out_dir, f'{prefix}_{name}')
            for ext in ('pdf', 'png'):
                path = f'{stem}.{ext}'
                self.fig.savefig(path, dpi=300, bbox_inches=bbox,
                                 facecolor=ts.SURFACE)
                written.append(path)
                print(f'  → {path}')
        return written


# ──────────────────────────────────────────────────────────────────────────────
# Einstiegspunkt
# ──────────────────────────────────────────────────────────────────────────────

def resolve_folder(argv) -> str:
    positional = [a for a in argv[1:] if not a.startswith('-')]
    if positional:
        folder = os.path.abspath(positional[0])
        if not os.path.isdir(folder):
            print(f'Folder not found: {folder}')
            sys.exit(1)
        return folder


    # No argument: prefer a folder configured in JOINTS, else the first one
    for joint_name in tss.JOINTS:
        cand = os.path.join(BUILDS_DIR, joint_name)
        if glob.glob(os.path.join(cand, 'spirob_messung_*.csv')):
            print(f'No folder given - using the JOINTS entry: {joint_name}')
            return cand

    folders = list_recording_folders()
    if not folders:
        print(f'No measurement folders with spirob_messung_*.csv under {BUILDS_DIR}')
        sys.exit(1)
    print(f'No folder given - using: {os.path.basename(folders[0])}')
    return folders[0]


def main():
    folder = resolve_folder(sys.argv)
    gui = SysIdGUI(folder)

    if '--plot' in sys.argv:
        print(f'\nSignal trace as a figure ({gui.folder_name}):')
        gui.export_signal_plot()
        return

    if '--figure' in sys.argv:
        # Output folder: the argument after --figure, else build/free_vibration/gui
        i = sys.argv.index('--figure')
        rest = [a for a in sys.argv[i + 1:] if not a.startswith('-')]
        out_dir = (os.path.abspath(rest[0]) if rest and os.path.isdir(rest[0]) is False
                   and not os.path.isdir(os.path.join(BUILDS_DIR, rest[0]))
                   else os.path.join(tss.RESULTS_DIR, 'gui'))
        prefix = f'gui_{gui.folder_name}'
        print(f'\nInterface figures ({gui.folder_name}):')
        gui.export_figures(out_dir, prefix=prefix)
        return

    gui.run()


if __name__ == '__main__':
    main()
