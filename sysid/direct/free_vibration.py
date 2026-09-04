"""Joint stiffness and damping from free-vibration ring-down (log decrement).

One segment is clamped so exactly one joint can swing, in a *horizontal* plane:
the joint axis stands vertical, so gravity acts parallel to the axis, adds no
restoring moment, and never has to be compensated. A three-axis accelerometer
sits at a known radius, with its y-axis along the tangential direction, so the
angular acceleration follows directly from the y signal.

The angle is never reconstructed by double integration -- noise and offset
would drift. Instead the ring-down is read straight off the acceleration, which
is legitimate because the second derivative of

    theta(t) = A e^(-D w0 t) cos(wd t + phi)

has the *same* envelope and the same damped frequency, differing only in
amplitude (by w0^2) and phase. The constant amplitude factor cancels inside the
logarithmic decrement, so:

    wd  from the spacing of successive extrema
    Lam from the amplitude ratio of successive extrema
    D   = Lam / sqrt(4 pi^2 + Lam^2)
    k   = (wd / sqrt(1 - D^2))^2 * J
    d   = 2 D wd / sqrt(1 - D^2) * J

Because k and d both scale linearly with J, they are only as good as the
moment of inertia, which is *hand-tuned per joint* in the GUI and stored in
each folder's ``sysid_settings.yaml``. D itself follows from the decrement
alone and is independent of J -- which is why it comes out near-constant
(0.12-0.13) across all four joints while k does not.

Usage::

    uv run sysid/direct/free_vibration.py           # batch over every joint
    uv run sysid/direct/free_vibration_gui.py       # interactive, writes the YAML

Note the batch run does *not* apply the per-file manual start points that only
exist in the GUI, so it can differ from the YAML ``results`` block by a few
percent. The YAML values are the reference: they are what seeds the real-to-sim
identification.

Inputs : data/free_vibration/joint_*/spirob_messung_*.csv + sysid_settings.yaml
Outputs: build/free_vibration/
"""

import csv
import glob
import math
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import butter, filtfilt, find_peaks

from spirob import plotstyle as ts
from spirob.paths import FREE_VIBRATION_DIR
from spirob.paths import build_dir as _build_dir

# ============================================================================
# CONFIGURATION
# ============================================================================

# Ring-down recordings: data/free_vibration/joint_NN/spirob_messung_*.csv
BUILDS_DIR = str(FREE_VIBRATION_DIR)

# Results go to build/, never next to the measurements.
RESULTS_DIR = str(_build_dir('free_vibration'))

# Configuration per joint folder
# - sensor_id: which sensor board to evaluate
# - J: moment of inertia of the swinging segment, in kg·m^2
#
# These are read from each folder's sysid_settings.yaml by discover_joints().
def discover_joints(root: str = BUILDS_DIR) -> dict:
    """Every ``joint_NN`` folder that carries a ``sysid_settings.yaml``.

    ``sensor_id`` and above all ``J`` (the swinging segment's moment of
    inertia) come from that YAML, because J is hand-tuned per joint in the GUI
    and k = omega_0^2 * J scales linearly with it -- reading it from anywhere
    else would silently produce different stiffnesses than the ones published.
    """
    found = {}
    for entry in sorted(os.listdir(root)):
        folder = os.path.join(root, entry)
        settings = os.path.join(folder, SETTINGS_FILENAME)
        if not os.path.isdir(folder) or not os.path.isfile(settings):
            continue
        with open(settings) as fh:
            doc = yaml.safe_load(fh) or {}
        cfg = doc.get('settings') or {}
        if 'J' not in cfg:
            print(f"  skipping {entry}: no 'settings.J' in {SETTINGS_FILENAME}")
            continue
        found[entry] = {'sensor_id': int(cfg.get('sensor_id', 0)),
                        'J': float(cfg['J']),
                        'settings': cfg}
    return found


def params_from_settings(cfg: dict) -> 'AnalysisParams':
    """Build AnalysisParams from a joint's ``sysid_settings.yaml`` settings block.

    Note what this does *not* carry over: the per-file manual start points
    (``start_overrides``). Those only exist in the GUI, so this batch run can
    differ from the YAML ``results`` block by a few percent. The YAML results
    are the reference values -- they are what seeds the real-to-sim run.
    """
    d = AnalysisParams()
    for field_name in ('lowpass_enabled', 'lowpass_cutoff_hz', 'lowpass_order',
                       'peak_prominence_factor', 'peak_min_distance_s',
                       'min_peaks_required', 'outlier_sigma', 'trigger_mode'):
        if field_name in cfg:
            setattr(d, field_name, cfg[field_name])
    return d


SETTINGS_FILENAME = 'sysid_settings.yaml'

# Filled in main() from the YAML files; override here to analyse a subset.
JOINTS: dict = {}

# Signalverarbeitung
LOWPASS_ENABLED = True       # Butterworth-Tiefpassfilter aktivieren
LOWPASS_CUTOFF_HZ = 50.0    # Grenzfrequenz in Hz
LOWPASS_ORDER = 4            # Filterordnung

# Peak-Erkennung
MIN_PEAKS_REQUIRED = 3       # fewest peaks a measurement needs to count
PEAK_PROMINENCE_FACTOR = 0.1 # Prominence = Faktor × max. Amplitude

# Outlier rejection
OUTLIER_SIGMA = 2.0          # measurements more than 2 sigma from the mean are flagged


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MeasurementResult:
    """Result of a single measurement."""
    filename: str
    joint_name: str
    sensor_id: int
    J: float
    omega_d: float          # damped natural angular frequency [rad/s]
    omega_n: float          # undamped natural angular frequency [rad/s]
    f_d: float              # damped natural frequency [Hz]
    f_n: float              # undamped natural frequency [Hz]
    zeta: float             # damping ratio [-]
    delta: float            # Logarithmisches Dekrement [-]
    k: float                # stiffness [N·m/rad]
    d: float                # damping [N·m·s/rad]
    n_peaks: int            # number of peaks used
    is_outlier: bool = False


@dataclass
class JointSummary:
    """Aggregate over every measurement of one joint."""
    joint_name: str
    n_measurements: int
    n_valid: int
    n_outliers: int
    omega_d_mean: float
    omega_d_std: float
    omega_n_mean: float
    omega_n_std: float
    zeta_mean: float
    zeta_std: float
    k_mean: float
    k_std: float
    d_mean: float
    d_std: float


@dataclass
class AnalysisParams:
    """Every tunable parameter of the ring-down analysis.

    Defaults mirror the module constants, so behaviour is unchanged when no
    parameters are passed.
    """
    lowpass_enabled: bool = LOWPASS_ENABLED
    lowpass_cutoff_hz: float = LOWPASS_CUTOFF_HZ
    lowpass_order: int = LOWPASS_ORDER
    peak_prominence_factor: float = PEAK_PROMINENCE_FACTOR
    peak_min_distance_s: float = 0.02
    min_peaks_required: int = MIN_PEAKS_REQUIRED
    outlier_sigma: float = OUTLIER_SIGMA
    trigger_mode: str = 'onset'          # 'onset' = start of oscillation, 'jump' = largest step
    onset_noise_factor: float = 4.0      # sensitivity of the onset detection


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def apply_lowpass_filter(signal: np.ndarray, fs: float,
                         cutoff: float = LOWPASS_CUTOFF_HZ,
                         order: int = LOWPASS_ORDER) -> np.ndarray:
    """Apply a Butterworth low-pass filter to the signal."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        # Cutoff above Nyquist -> no filtering
        return signal
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)


def estimate_sample_rate(t_s: np.ndarray) -> float:
    """Estimate the sample rate from the timestamps."""
    dt = np.diff(t_s)
    dt_positive = dt[dt > 0]
    if len(dt_positive) == 0:
        return 500.0  # Fallback
    return 1.0 / np.median(dt_positive)


# ============================================================================
# DATEN LADEN
# ============================================================================

def load_csv(filepath: str, sensor_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV file and filter it by sensor_id.

    Returns:
        t_s: Zeitachse in Sekunden (relativ, ab 0 = Trigger)
        signal: accY values (the evaluated signal; the YZ sum is no longer used)
    """
    t_us_list = []
    signal_list = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'accY' not in fieldnames:
            raise ValueError(f"No 'accY' column in {filepath}")
        for row in reader:
            if int(row['sensor_id']) == sensor_id:
                t_us_list.append(int(row['t_us']))
                signal_list.append(float(row['accY']))

    if len(t_us_list) == 0:
        raise ValueError(f"No data for sensor_id={sensor_id} in {filepath}")

    t_us = np.array(t_us_list, dtype=np.float64)
    signal = np.array(signal_list, dtype=np.float64)

    # Zeitachse normalisieren: Erster Zeitstempel = 0
    t_s = (t_us - t_us[0]) / 1_000_000.0

    return t_s, signal


def find_trigger_index(t_s: np.ndarray, signal: np.ndarray) -> int:
    """
    Find the trigger instant as the largest step in the signal.

    The recording script stores pre-trigger data (about 0.5 s before the trigger).
    """
    if len(signal) < 10:
        return 0

    # Absolute difference to the previous sample
    diff = np.abs(np.diff(signal))

    # Largest step = trigger
    trigger_idx = np.argmax(diff)

    return trigger_idx


def find_onset_index(t_s: np.ndarray, signal: np.ndarray,
                     noise_factor: float = 4.0) -> int:
    """
    Find where the oscillation begins (rather than the largest step).

    Takes a quiet baseline from the start of the signal and returns the first
    index where the signal departs from it clearly (noise_factor * sigma_noise,
    or at least 5 % of the peak amplitude). This also captures the first few
    oscillations, which lie before the largest step.
    """
    n = len(signal)
    if n < 10:
        return 0

    # Baseline from the quiet start -- at most up to the largest step
    jump = find_trigger_index(t_s, signal)
    n_base = max(5, min(int(jump), int(n * 0.2)))
    baseline = signal[:n_base]
    mean = float(np.mean(baseline))
    noise = float(np.std(baseline))

    centered = np.abs(signal - mean)
    thresh = max(noise_factor * noise, 0.05 * float(np.max(centered)))

    above = np.where(centered > thresh)[0]
    if len(above) == 0:
        return jump
    return int(above[0])


# ============================================================================
# KERN-ALGORITHMUS: LOGARITHMISCHES DEKREMENT
# ============================================================================

def analyze_free_vibration(t_s: np.ndarray, signal: np.ndarray,
                           fs: float,
                           params: Optional['AnalysisParams'] = None,
                           trigger_idx: int | None = None) -> dict | None:
    """
    Analyse one free ring-down.

    Args:
        params: analysis parameters. None -> the module constants.
        trigger_idx: Fester Startindex (manuelle Vorgabe). None → automatische
            Detection follows ``params.trigger_mode`` ('onset' or 'jump').

    Returns:
        dict with omega_d, zeta, delta, n_peaks, peak_times, peak_amplitudes
        or None if not enough peaks were found
    """
    if params is None:
        params = AnalysisParams()

    # --- 1. Find the trigger/onset and centre the time axis ---
    if trigger_idx is None:
        if params.trigger_mode == 'onset':
            trigger_idx = find_onset_index(t_s, signal, params.onset_noise_factor)
        else:
            trigger_idx = find_trigger_index(t_s, signal)
    else:
        trigger_idx = int(np.clip(trigger_idx, 0, len(signal) - 1))

    # Pre-trigger window used to estimate the offset
    pre_trigger = signal[:max(trigger_idx, 1)]
    offset = np.mean(pre_trigger) if len(pre_trigger) > 5 else 0.0

    # Signal zentrieren
    signal_centered = signal - offset

    # Use only the post-trigger part (from the trigger index onwards)
    t_post = t_s[trigger_idx:] - t_s[trigger_idx]
    sig_post = signal_centered[trigger_idx:]

    if len(sig_post) < 20:
        return None

    # --- 2. Optionaler Tiefpassfilter ---
    if params.lowpass_enabled:
        sig_filtered = apply_lowpass_filter(sig_post, fs,
                                            params.lowpass_cutoff_hz,
                                            params.lowpass_order)
    else:
        sig_filtered = sig_post.copy()

    # --- 3. Peaks finden (Maxima) ---
    max_amp = np.max(np.abs(sig_filtered))
    if max_amp < 1e-6:
        return None

    prominence = params.peak_prominence_factor * max_amp
    # Minimum distance between peaks, in samples
    min_distance = max(5, int(fs * params.peak_min_distance_s))

    # Positive Peaks (Maxima)
    pos_peak_idx, pos_props = find_peaks(
        sig_filtered,
        prominence=prominence,
        distance=min_distance
    )

    # Negative Peaks (Minima) → invertiertes Signal
    neg_peak_idx, neg_props = find_peaks(
        -sig_filtered,
        prominence=prominence,
        distance=min_distance
    )

    # Merge all peaks, sorted by time
    all_peak_idx = np.sort(np.concatenate([pos_peak_idx, neg_peak_idx]))
    all_peak_amplitudes = np.abs(sig_filtered[all_peak_idx])
    all_peak_times = t_post[all_peak_idx]

    if len(all_peak_idx) < params.min_peaks_required:
        return None

    # --- 4. Damped natural frequency w_d ---
    # Successive peaks of equal polarity span one full period
    # Or from all peaks -> half periods
    # Using all peaks (half periods) gives better statistics
    half_periods = np.diff(all_peak_times)
    half_periods = half_periods[half_periods > 0]

    if len(half_periods) == 0:
        return None

    # Full period = 2 x median of the half periods
    T_d = 2.0 * np.median(half_periods)
    omega_d = 2.0 * np.pi / T_d

    # --- 5. Logarithmisches Dekrement δ ---
    # Successive peaks of equal polarity = one full period,
    # i.e. take every 2nd peak (all even OR all odd)
    # Better: only positive peaks OR only negative peaks
    pos_amplitudes = sig_filtered[pos_peak_idx]
    neg_amplitudes = np.abs(sig_filtered[neg_peak_idx])

    # Keep whichever series has more peaks
    if len(pos_amplitudes) >= len(neg_amplitudes):
        peak_amps_series = np.abs(pos_amplitudes)
    else:
        peak_amps_series = neg_amplitudes

    if len(peak_amps_series) < 2:
        return None

    # Logarithmic decrement: delta = ln(A_i / A_{i+1}) over successive peaks
    deltas = []
    for i in range(len(peak_amps_series) - 1):
        A_i = peak_amps_series[i]
        A_next = peak_amps_series[i + 1]
        if A_next > 1e-9 and A_i > A_next:  # only decaying pairs
            deltas.append(math.log(A_i / A_next))

    if len(deltas) == 0:
        # Fallback: over all peaks (factor 2, because these are half periods)
        for i in range(len(all_peak_amplitudes) - 2):
            A_i = all_peak_amplitudes[i]
            A_next = all_peak_amplitudes[i + 2]  # two ahead = same polarity
            if A_next > 1e-9 and A_i > A_next:
                deltas.append(math.log(A_i / A_next))

    if len(deltas) == 0:
        # Last resort: overall decay rate across all peaks
        if len(all_peak_amplitudes) >= 2:
            A_first = all_peak_amplitudes[0]
            A_last = all_peak_amplitudes[-1]
            n_half_cycles = len(all_peak_amplitudes) - 1
            if A_last > 1e-9 and A_first > A_last:
                # Over n half cycles: delta_total = ln(A_first/A_last)
                # Pro vollen Zyklus: δ = δ_gesamt / (n_half_cycles / 2)
                delta_total = math.log(A_first / A_last)
                delta = delta_total / (n_half_cycles / 2.0)
                deltas = [delta]

    if len(deltas) == 0:
        return None

    delta = np.mean(deltas)

    # --- 6. Damping ratio zeta ---
    zeta = delta / math.sqrt(4.0 * math.pi**2 + delta**2)

    return {
        'omega_d': omega_d,
        'zeta': zeta,
        'delta': delta,
        'n_peaks': len(all_peak_idx),
        # For the plots:
        't_post': t_post,
        'sig_post': sig_post,
        'sig_filtered': sig_filtered,
        'all_peak_idx': all_peak_idx,
        'all_peak_times': all_peak_times,
        'all_peak_amplitudes': all_peak_amplitudes,
        'pos_peak_idx': pos_peak_idx,
        'neg_peak_idx': neg_peak_idx,
        'T_d': T_d,
        'offset': offset,
        'trigger_idx': trigger_idx,
    }


# ============================================================================
# RESULT COMPUTATION
# ============================================================================

def compute_parameters(omega_d: float, zeta: float, J: float) -> dict:
    """Compute w_n, k and d from w_d, zeta and J."""
    # ω_n = ω_d / sqrt(1 - ζ²)
    if zeta >= 1.0:
        # Overdamped -- should not happen in a free-vibration test
        omega_n = omega_d
    else:
        omega_n = omega_d / math.sqrt(1.0 - zeta**2)

    k = J * omega_n**2
    d = 2.0 * J * zeta * omega_n

    f_d = omega_d / (2.0 * math.pi)
    f_n = omega_n / (2.0 * math.pi)

    return {
        'omega_n': omega_n,
        'f_d': f_d,
        'f_n': f_n,
        'k': k,
        'd': d,
    }


# ============================================================================
# DIAGNOSE-PLOTS
# ============================================================================

def create_diagnostic_plot(result: MeasurementResult, analysis: dict,
                           output_path: str):
    """Diagnostic plot for a single measurement.

    ``output_path`` is treated as a file stem; written are PDF
    (for the document) and PNG (for quick viewing).
    """
    ts.apply_style()

    fig, axes = plt.subplots(2, 1, figsize=(ts.FIG_FULL, 4.8),
                             height_ratios=[3, 1.15],
                             gridspec_kw={'hspace': 0.42})

    t = analysis['t_post']
    sig_raw = analysis['sig_post']
    sig_filt = analysis['sig_filtered']
    peak_idx = analysis['all_peak_idx']

    # --- Subplot 1: signal with peaks and envelope ---
    ax1 = axes[0]
    ax1.axhline(y=0, color=ts.BASELINE, linewidth=0.8, zorder=2)
    ax1.axvline(x=0, color=ts.MUTED, linewidth=0.9, linestyle=(0, (1, 1.6)),
                zorder=2, label='Trigger')

    ax1.plot(t, sig_raw, color=ts.FAINT, linewidth=0.7, label='Rohsignal', zorder=3)
    ax1.plot(t, sig_filt, label='Gefiltert', zorder=4, **ts.line_kw(0, width=1.2))

    # Mark the peaks -- shape carries the distinction, not colour alone
    ax1.plot(t[analysis['pos_peak_idx']], sig_filt[analysis['pos_peak_idx']],
             label='Maxima', zorder=6, **ts.marker_kw(1, size=5.5, marker='v'))
    ax1.plot(t[analysis['neg_peak_idx']], sig_filt[analysis['neg_peak_idx']],
             label='Minima', zorder=6, **ts.marker_kw(2, size=5.5, marker='^'))

    # Envelope fitted through the peaks -- neutral in colour so it stays
    # stands out from every series colour in greyscale print
    decay_rate = 0.0
    if len(peak_idx) >= 2:
        peak_a = np.abs(sig_filt[peak_idx])
        try:
            if result.zeta > 0 and result.omega_n > 0:
                decay_rate = result.zeta * result.omega_n
                t_envelope = np.linspace(0, t[-1], 800)
                A0 = peak_a[0]
                envelope = A0 * np.exp(-decay_rate * t_envelope)
                ax1.plot(t_envelope, envelope,
                         label=f'Envelope ($\\zeta\\omega_n$ = {ts.num(decay_rate, 1)} 1/s)',
                         **ts.model_kw(width=1.2))
                ax1.plot(t_envelope, -envelope, **ts.model_kw(width=1.2))
        except Exception:
            decay_rate = 0.0

    ts.grid_on(ax1)
    ax1.set_title(f"{result.joint_name} – {os.path.basename(result.filename)}", pad=8)
    ax1.set_xlabel('Time $t$ after trigger (s)')
    ax1.set_ylabel('Acceleration (g)')

    # Zoom onto the ring-down window -- the long flat tail after it adds
    # nothing and squeezes the oscillation into a few millimetres in print.
    t_end = t[-1]
    if decay_rate > 0:
        t_end = min(t[-1], max(3.0 / decay_rate, t[peak_idx[-1]] * 3.0, 0.05))
    ax1.set_xlim(t[0], t_end)

    window = (t >= t[0]) & (t <= t_end)
    span = float(np.nanmax(np.abs(sig_raw[window]))) if window.any() else 1.0
    ax1.set_ylim(-span * 1.08, span * 1.08)

    ax1.legend(loc='upper right', ncol=2, columnspacing=1.2)

    ts.annotate(ax1, "  ".join([
        f"$\\omega_d$ = {ts.de_num(result.omega_d, 1)} rad/s",
        f"$\\zeta$ = {ts.de_num(result.zeta, 4)}",
        f"$k$ = {ts.de_auto(result.k)} N·m/rad",
        f"$d$ = {ts.de_auto(result.d)} N·m·s/rad",
    ]), loc='lower right', fontsize=7.5)

    # --- Subplot 2: Peak-Amplituden (Abklingkurve) ---
    ax2 = axes[1]
    if len(peak_idx) >= 2:
        peak_numbers = np.arange(len(peak_idx))
        peak_amps = np.abs(sig_filt[peak_idx])
        ax2.bar(peak_numbers, peak_amps, width=0.7, **ts.bar_kw(0))
        ax2.set_xlabel('Peak-Nummer $i$')
        ax2.set_ylabel('$|A_i|$ (g)')
        ax2.set_xticks(peak_numbers[::max(1, len(peak_numbers) // 12)])
        ts.grid_on(ax2, axis='y')
        ts.annotate(ax2, f"$\\delta$ = {ts.de_num(result.delta, 4)},  "
                         f"$n$ = {result.n_peaks} Peaks",
                    loc='upper right', fontsize=7.5)

    ts.german_axes(fig)
    ts.save(fig, output_path, quiet=True)


# ============================================================================
# SUMMARY
# ============================================================================

def compute_joint_summary(joint_name: str,
                          results: list[MeasurementResult]) -> JointSummary | None:
    """Aggregate every measurement of one joint."""
    if not results:
        return None

    valid = [r for r in results if not r.is_outlier]

    if not valid:
        return JointSummary(
            joint_name=joint_name,
            n_measurements=len(results),
            n_valid=0, n_outliers=len(results),
            omega_d_mean=0, omega_d_std=0,
            omega_n_mean=0, omega_n_std=0,
            zeta_mean=0, zeta_std=0,
            k_mean=0, k_std=0,
            d_mean=0, d_std=0,
        )

    omega_d = np.array([r.omega_d for r in valid])
    omega_n = np.array([r.omega_n for r in valid])
    zeta = np.array([r.zeta for r in valid])
    k = np.array([r.k for r in valid])
    d = np.array([r.d for r in valid])

    return JointSummary(
        joint_name=joint_name,
        n_measurements=len(results),
        n_valid=len(valid),
        n_outliers=len(results) - len(valid),
        omega_d_mean=float(np.mean(omega_d)),
        omega_d_std=float(np.std(omega_d)),
        omega_n_mean=float(np.mean(omega_n)),
        omega_n_std=float(np.std(omega_n)),
        zeta_mean=float(np.mean(zeta)),
        zeta_std=float(np.std(zeta)),
        k_mean=float(np.mean(k)),
        k_std=float(np.std(k)),
        d_mean=float(np.mean(d)),
        d_std=float(np.std(d)),
    )


def mark_outliers(results: list[MeasurementResult],
                  sigma: float = OUTLIER_SIGMA) -> list[MeasurementResult]:
    """Flag measurements whose w_d deviates by more than sigma standard deviations."""
    if len(results) < 3:
        return results

    omega_d_values = np.array([r.omega_d for r in results])
    mean = np.mean(omega_d_values)
    std = np.std(omega_d_values)

    if std < 1e-9:
        return results

    for r in results:
        if abs(r.omega_d - mean) > sigma * std:
            r.is_outlier = True

    return results


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_results_csv(all_results: list[MeasurementResult],
                       summaries: list[JointSummary],
                       output_dir: str):
    """Export every result as CSV files."""
    # --- Einzelergebnisse ---
    detail_path = os.path.join(output_dir, 'free_vibration_per_measurement.csv')
    headers = [
        'joint', 'file', 'sensor_id', 'J_kgm2',
        'omega_d_rad_s', 'omega_n_rad_s', 'f_d_Hz', 'f_n_Hz',
        'zeta', 'delta', 'k_Nm_rad', 'd_Nms_rad',
        'n_peaks', 'ist_ausreisser'
    ]

    with open(detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in all_results:
            writer.writerow([
                r.joint_name, os.path.basename(r.filename), r.sensor_id, r.J,
                f'{r.omega_d:.4f}', f'{r.omega_n:.4f}', f'{r.f_d:.4f}', f'{r.f_n:.4f}',
                f'{r.zeta:.6f}', f'{r.delta:.6f}', f'{r.k:.8f}', f'{r.d:.8f}',
                r.n_peaks, r.is_outlier
            ])

    print(f"  -> per-measurement CSV: {detail_path}")

    # --- Summary ---
    summary_path = os.path.join(output_dir, 'free_vibration_summary.csv')
    headers = [
        'joint', 'n_measurements', 'n_valid', 'n_outliers',
        'omega_d_mean', 'omega_d_std',
        'omega_n_mean', 'omega_n_std',
        'zeta_mean', 'zeta_std',
        'k_mean_Nm_rad', 'k_std',
        'd_mean_Nms_rad', 'd_std'
    ]

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in summaries:
            writer.writerow([
                s.joint_name, s.n_measurements, s.n_valid, s.n_outliers,
                f'{s.omega_d_mean:.4f}', f'{s.omega_d_std:.4f}',
                f'{s.omega_n_mean:.4f}', f'{s.omega_n_std:.4f}',
                f'{s.zeta_mean:.6f}', f'{s.zeta_std:.6f}',
                f'{s.k_mean:.8f}', f'{s.k_std:.8f}',
                f'{s.d_mean:.8f}', f'{s.d_std:.8f}',
            ])

    print(f"  -> summary CSV:         {summary_path}")


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def print_results_table(results: list[MeasurementResult], joint_name: str):
    """Print a formatted result table."""
    print(f"\n{'─' * 120}")
    print(f"  Joint: {joint_name}")
    print(f"{'─' * 120}")
    print(f"  {'File':<40} {'ω_d [rad/s]':>12} {'ω_n [rad/s]':>12} "
          f"{'ζ':>10} {'k [Nm/rad]':>14} {'d [Nm·s/rad]':>14} {'Peaks':>6} {'Status':>10}")
    print(f"  {'─' * 118}")

    for r in results:
        status = "OUTLIER" if r.is_outlier else "ok"
        basename = os.path.basename(r.filename)
        print(f"  {basename:<40} {r.omega_d:>12.4f} {r.omega_n:>12.4f} "
              f"{r.zeta:>10.6f} {r.k:>14.8f} {r.d:>14.8f} {r.n_peaks:>6} {status:>10}")


def print_summary(summary: JointSummary):
    """Print the summary of one joint."""
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  SUMMARY: {summary.joint_name:<51}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  measurements:      {summary.n_measurements:<5}  "
          f"valid: {summary.n_valid:<5}  outliers: {summary.n_outliers:<5}  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  ω_d  = {summary.omega_d_mean:>10.4f} ± {summary.omega_d_std:<10.4f} rad/s           ║")
    print(f"  ║  ω_n  = {summary.omega_n_mean:>10.4f} ± {summary.omega_n_std:<10.4f} rad/s           ║")
    print(f"  ║  ζ    = {summary.zeta_mean:>10.6f} ± {summary.zeta_std:<10.6f}                 ║")
    print(f"  ║  k    = {summary.k_mean:>10.8f} ± {summary.k_std:<10.8f} Nm/rad       ║")
    print(f"  ║  d    = {summary.d_mean:>10.8f} ± {summary.d_std:<10.8f} Nm·s/rad     ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")


# ============================================================================
# SUMMARYS-PLOT
# ============================================================================

def _joint_label(name: str) -> str:
    """``joint_11`` -> ``11``. Short, because the axis is already labelled 'Joint'."""
    tail = name.replace('joint_', '').replace('data_', '').lstrip('0')
    return tail if tail.isdigit() else name


def create_summary_plot(summaries: list[JointSummary], output_dir: str):
    """Comparison plot across all joints (PDF + PNG)."""
    if not summaries or all(s.n_valid == 0 for s in summaries):
        return

    valid_summaries = [s for s in summaries if s.n_valid > 0]
    if not valid_summaries:
        return

    ts.apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(ts.FIG_FULL, 3.0))

    names = [_joint_label(s.joint_name) for s in valid_summaries]
    x = np.arange(len(names), dtype=float)
    width = 0.55

    panels = [
        # (slot, values, spread, axis label, panel title, formatter)
        (0, [s.k_mean for s in valid_summaries], [s.k_std for s in valid_summaries],
         '$k$ (N·m/rad)', '(a) Stiffness', ts.auto),
        (1, [s.d_mean for s in valid_summaries], [s.d_std for s in valid_summaries],
         '$d$ (N·m·s/rad)', '(b) Damping', ts.auto),
        (2, [s.zeta_mean for s in valid_summaries], [s.zeta_std for s in valid_summaries],
         '$\\zeta$ (–)', '(c) Damping ratio', ts.auto),
    ]

    for ax, (slot, means, stds, ylabel, title, fmt) in zip(axes, panels):
        ax.bar(x, means, width, yerr=stds, **ts.bar_kw(slot, hatch=None),
               error_kw=dict(ecolor=ts.INK_2, elinewidth=0.9, capsize=3,
                             capthick=0.9, zorder=4))
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_xlabel('Joint')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ts.grid_on(ax, axis='y')

        # Headroom for the direct value labels
        top = max((m + s) for m, s in zip(means, stds)) or 1.0
        ax.set_ylim(0, top * 1.26)
        ts.value_labels(ax, x, [m + s for m, s in zip(means, stds)],
                        [fmt(m) for m in means])

    fig.suptitle('Torsion-joint system identification — mean ± standard deviation',
                 y=1.03)
    fig.tight_layout()

    # The x-axes are categorical text already -- only reformat the y-axes
    ts.localize_axes(fig, skip_x=list(axes))

    plot_path = os.path.join(output_dir, 'free_vibration_summary')
    print("\n  -> summary plot:")
    ts.save(fig, plot_path)


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def process_joint(joint_name: str, config: dict,
                  results_dir: str,
                  analysis_params: AnalysisParams | None = None) -> list[MeasurementResult]:
    """Process every measurement of one joint."""
    if analysis_params is None:
        analysis_params = AnalysisParams()
    sensor_id = config['sensor_id']
    J = config['J']

    joint_dir = os.path.join(BUILDS_DIR, joint_name)
    if not os.path.isdir(joint_dir):
        print(f"\n  {joint_dir} does not exist - skipping {joint_name}")
        return []

    csv_files = sorted(glob.glob(os.path.join(joint_dir, 'spirob_messung_*.csv')))
    if not csv_files:
        print(f"\n  no CSV files in {joint_dir} - skipping {joint_name}")
        return []

    # Per-joint output folder for the plots
    joint_results_dir = os.path.join(results_dir, joint_name)
    os.makedirs(joint_results_dir, exist_ok=True)

    print(f"\n{'═' * 120}")
    print(f"  JOINT: {joint_name}  |  sensor id: {sensor_id}  |  J = {J} kg·m^2  |  {len(csv_files)} measurements")
    print(f"{'═' * 120}")

    results = []

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)

        try:
            # Daten laden
            t_s, signal = load_csv(csv_file, sensor_id)

            # Estimate the sample rate
            fs = estimate_sample_rate(t_s)

            # Ausschwingvorgang analysieren
            analysis = analyze_free_vibration(t_s, signal, fs, analysis_params)

            if analysis is None:
                print(f"  {basename}: not enough peaks found - skipped")
                continue

            # Parameter berechnen
            params = compute_parameters(analysis['omega_d'], analysis['zeta'], J)

            # Build the result
            result = MeasurementResult(
                filename=csv_file,
                joint_name=joint_name,
                sensor_id=sensor_id,
                J=J,
                omega_d=analysis['omega_d'],
                omega_n=params['omega_n'],
                f_d=params['f_d'],
                f_n=params['f_n'],
                zeta=analysis['zeta'],
                delta=analysis['delta'],
                k=params['k'],
                d=params['d'],
                n_peaks=analysis['n_peaks'],
            )
            results.append(result)

            # Diagnoseplot erstellen
            plot_name = basename.replace('.csv', '_sysid')
            plot_path = os.path.join(joint_results_dir, plot_name)
            create_diagnostic_plot(result, analysis, plot_path)

        except Exception as e:
            print(f"  {basename}: error - {e}")

    if results:
        # Flag outliers
        results = mark_outliers(results, analysis_params.outlier_sigma)

        # Tabelle ausgeben
        print_results_table(results, joint_name)

    return results


def main():
    """Entry point."""
    print("\n" + "█" * 120)
    print("  SpiRob torsion spring - automated system identification")
    print("  Method: logarithmic decrement of free ring-down measurements")
    print("█" * 120)

    # Create the results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results: list[MeasurementResult] = []
    summaries: list[JointSummary] = []

    params = AnalysisParams()

    joints = JOINTS or discover_joints()
    if not joints:
        print(f"\n  No joint folders with {SETTINGS_FILENAME} found under {BUILDS_DIR}")
        return

    for joint_name, config in joints.items():
        joint_params = params_from_settings(config['settings']) if 'settings' in config else params
        results = process_joint(joint_name, config, RESULTS_DIR, joint_params)
        all_results.extend(results)

        summary = compute_joint_summary(joint_name, results)
        if summary:
            summaries.append(summary)
            print_summary(summary)

    # CSV-Export
    if all_results:
        print(f"\n{'═' * 120}")
        print("  EXPORT")
        print(f"{'═' * 120}")
        export_results_csv(all_results, summaries, RESULTS_DIR)
        create_summary_plot(summaries, RESULTS_DIR)

    # Abschluss
    print(f"\n{'█' * 120}")
    print(f"  Done. {len(all_results)} measurements evaluated.")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'█' * 120}\n")


if __name__ == '__main__':
    main()
