#!/usr/bin/env python3
"""
SpiRob Live Dashboard — MuJoCo GUI + live dynamics plots

Opens the interactive MuJoCo viewer with the *identified* joint/tendon
parameters (from ``sysid_real_modular.py``) already applied, so you can drive
the robot with the actuator sliders in the viewer's Control panel and watch,
in real time, everything MuJoCo exposes:

  * joint angles  q      (mjData.qpos)              -- time plot
  * Geschwindigkeiten v  (mjData.qvel)              — time plot
  * tip position (seg_0) in the x-z plane           -- 2-D trajectory
  * Massenmatrix M(q)    (mjData.qM → dense)        — heatmap
  * bias force c(q,v)    (mjData.qfrc_bias:         -- bar
                          Coriolis + Zentrifugal + Gravitation)
  * applied force tau    (qfrc_passive + _actuator  -- bar
                          + _applied)
  * constraint forces f  (mjData.efc_force, nefc)   -- bar

The **realtime factor (RTF)** is shown in the dashboard title.

PERFORMANCE / THREADING
-----------------------
Rendering a full Matplotlib frame costs ~60-80 ms. If the physics were stepped
in the same loop, every draw would stall the simulation (that was the old
slow-down). Instead:

  * The physics + MuJoCo viewer run in a background thread with *catch-up*
    pacing: the sim tracks the wall clock and — since a step is ~475× faster
    than realtime — instantly sprints through any backlog a draw caused. So the
    RTF stays ≈ 1.0 no matter how heavy the plots are.
  * Matplotlib runs in the main thread and reads a short, lock-guarded snapshot
    of the state, so the expensive draw is off the physics critical path.

NOTE on the "Coriolis matrix": MuJoCo does not expose a separate C(q,v) matrix.
It provides the combined *bias force vector* ``qfrc_bias`` = c(q,v) = Coriolis +
centrifugal + gravity (length nv), which is what the dashboard shows. The full
inertia matrix M(q) is shown as a heatmap.

Controls:
  * Drag the actuator sliders in the MuJoCo Control panel (ctrlrange [-50, 0],
    negative = tendon tension) to steer the robot.
  * Press 'R' in the MuJoCo window to reset the state.

Run:
    uv run apps/sys_id/spirob_live_dashboard.py
    uv run apps/sys_id/spirob_live_dashboard.py --params build/sysid_real_modular_validate.json
    uv run sysid/tools/live_dashboard.py --xml models/spirob_13seg.xml --no-params
    uv run apps/sys_id/spirob_live_dashboard.py --plot-hz 30 --matrix-every 3   # smoother plots
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np

from spirob.paths import DEFAULT_MODEL as DEFAULT_XML
from spirob.paths import DEFAULT_PARAMS, REPO_ROOT

TCP_BODY = "seg_0"        # topmost segment = tip = TCP (base → seg_13 → … → seg_0)
MAX_CATCHUP_STEPS = 60    # cap sim backlog caught up per physics iteration (~0.24 s @ dt=0.004)


# ── Parameter loading ────────────────────────────────────────────────────────

def apply_identified_params(model: mj.MjModel, params_path: Path) -> str:
    """Apply the identified per-joint / per-tendon parameters onto the model.

    Reads the flat arrays written by ``sysid_real_modular.write_params_json``
    (already in MuJoCo model-joint-index order) and writes them exactly the way
    ``sysid_real_modular.set_params`` does:
        stiffness    → jnt_stiffness[i]
        damping      → dof_damping[dofadr[i]]
        frictionloss → dof_frictionloss[dofadr[i]]
        tendon_*     → tendon_*[t]
    Returns a short human-readable summary string.
    """
    with open(params_path) as f:
        jd = json.load(f)

    st = jd.get("stiffness")
    dp = jd.get("damping")
    fl = jd.get("frictionloss")
    for i in range(model.njnt):
        dof = int(model.jnt_dofadr[i])
        if st is not None:
            model.jnt_stiffness[i] = float(st[i])
        if dp is not None:
            model.dof_damping[dof] = float(dp[i])
        if fl is not None:
            model.dof_frictionloss[dof] = float(fl[i])

    tst = jd.get("tendon_stiffness")
    tdp = jd.get("tendon_damping")
    tfl = jd.get("tendon_frictionloss")
    for t in range(model.ntendon):
        if tst is not None:
            model.tendon_stiffness[t] = float(tst[t])
        if tdp is not None:
            model.tendon_damping[t] = float(tdp[t])
        if tfl is not None:
            model.tendon_frictionloss[t] = float(tfl[t])

    return (
        f"{params_path.name} (mode={jd.get('mode', '?')}, "
        f"cost={jd.get('cost', float('nan')):.4f})"
    )


# ── Shared state between physics thread and plotting thread ──────────────────

class SharedState:
    """Thread-safe hand-off between the physics thread and the plot thread."""

    def __init__(self, model: mj.MjModel, data: mj.MjData):
        self.model = model
        self.data = data
        self.nv = model.nv
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.reset_flag = threading.Event()
        self.rtf = 0.0
        self.tcp_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, TCP_BODY)
        if self.tcp_bid < 0:
            self.tcp_bid = model.nbody - 1

    def snapshot(self) -> dict:
        """Consistent copy of everything the dashboard needs.

        The lock is held only for the cheap copies (a few nv-vectors + qM), so
        the physics thread is blocked ~1 ms per plot frame. The dense mass
        matrix is expanded from the copied qM *outside* the lock.
        """
        m = self.model
        d = self.data
        with self.lock:
            t = float(d.time)
            q = d.qpos.copy()
            v = d.qvel.copy()
            qM = d.qM.copy()
            bias = d.qfrc_bias.copy()
            tau = (d.qfrc_passive + d.qfrc_actuator + d.qfrc_applied).copy()
            tcp = d.xpos[self.tcp_bid].copy()
            nefc = int(d.nefc)
            efc = d.efc_force[:nefc].copy()
        M = np.zeros((self.nv, self.nv), dtype=np.float64)
        mj.mj_fullM(m, M, qM)
        return {"t": t, "q": q, "v": v, "M": M, "bias": bias,
                "tau": tau, "tcp": tcp, "nefc": nefc, "efc": efc, "rtf": self.rtf}


def physics_thread(shared: SharedState, init_ctrl: float) -> None:
    """Step the sim in real time (catch-up pacing) and run the MuJoCo viewer.

    Catch-up pacing: each iteration advances the sim by the wall-clock time that
    elapsed, sprinting through any backlog a heavy draw introduced. Because a
    step is ~475× faster than realtime, the RTF stays ≈ 1.0 regardless of plot
    cost. The backlog is capped (MAX_CATCHUP_STEPS) to avoid a spiral of death.
    """
    model, data = shared.model, shared.data
    dt = model.opt.timestep

    def key_callback(keycode: int) -> None:
        if keycode in (ord("R"), ord("r")):
            with shared.lock:
                mj.mj_resetData(model, data)
                for i in range(model.nu):
                    data.ctrl[i] = -abs(init_ctrl)
                mj.mj_forward(model, data)
            shared.reset_flag.set()
            print("[reset] state reset")

    accum = 0.0
    prev = time.perf_counter()
    last_sync = 0.0
    last_rtf_t = prev
    steps_total = 0
    steps_at_rtf = 0

    try:
        with mj_viewer.launch_passive(model, data, key_callback=key_callback) as v:
            while v.is_running() and not shared.stop.is_set():
                now = time.perf_counter()
                accum += now - prev
                prev = now

                n_steps = int(accum / dt)
                if n_steps > MAX_CATCHUP_STEPS:
                    n_steps = MAX_CATCHUP_STEPS
                    accum = 0.0            # drop backlog after a long stall
                else:
                    accum -= n_steps * dt

                if n_steps > 0:
                    with shared.lock:
                        for _ in range(n_steps):
                            mj.mj_step(model, data)
                    steps_total += n_steps

                if now - last_sync >= 1.0 / 60.0:
                    v.sync()
                    last_sync = now

                if now - last_rtf_t >= 0.5:
                    shared.rtf = (steps_total - steps_at_rtf) * dt / (now - last_rtf_t)
                    last_rtf_t = now
                    steps_at_rtf = steps_total

                time.sleep(0.001)
    finally:
        shared.stop.set()


# ── Dashboard (main thread) ──────────────────────────────────────────────────

class LiveDashboard:
    """A single Matplotlib window with all live plots, updated in-place."""

    def __init__(self, model: mj.MjModel, window_s: float, matrix_every: int):
        self.nv = model.nv
        self.window_s = window_s
        self.matrix_every = max(1, matrix_every)
        self._frame = 0

        self.t_hist: deque[float] = deque()
        self.q_hist: deque[np.ndarray] = deque()
        self.v_hist: deque[np.ndarray] = deque()
        self.tcp_hist: deque[tuple[float, float]] = deque(maxlen=4000)

        plt.ion()
        self.fig = plt.figure("SpiRob Live Dashboard", figsize=(16, 9))
        self.suptitle = self.fig.suptitle("Realtime factor: — ", fontsize=13, fontweight="bold")
        gs = self.fig.add_gridspec(3, 3, hspace=0.5, wspace=0.30, top=0.92)

        self.ax_q = self.fig.add_subplot(gs[0, 0:2])
        self.ax_tcp = self.fig.add_subplot(gs[0, 2])
        self.ax_v = self.fig.add_subplot(gs[1, 0:2])
        self.ax_M = self.fig.add_subplot(gs[1, 2])
        self.ax_bias = self.fig.add_subplot(gs[2, 0])
        self.ax_tau = self.fig.add_subplot(gs[2, 1])
        self.ax_efc = self.fig.add_subplot(gs[2, 2])

        colors = plt.cm.tab20(np.linspace(0, 1, max(self.nv, 2)))
        self.q_lines = [self.ax_q.plot([], [], color=colors[i], lw=1.2)[0] for i in range(self.nv)]
        self.v_lines = [self.ax_v.plot([], [], color=colors[i], lw=1.2)[0] for i in range(self.nv)]
        self.ax_q.set_title("Joint angles q  (qpos)")
        self.ax_q.set_ylabel("angle (rad)")
        self.ax_q.grid(True, ls="--", alpha=0.4)
        self.ax_v.set_title("Joint velocities v  (qvel)")
        self.ax_v.set_xlabel("time (s)")
        self.ax_v.set_ylabel("vel (rad/s)")
        self.ax_v.grid(True, ls="--", alpha=0.4)

        (self.tcp_trace,) = self.ax_tcp.plot([], [], color="steelblue", lw=1.0, alpha=0.7)
        (self.tcp_dot,) = self.ax_tcp.plot([], [], "o", color="crimson", ms=7)
        self.ax_tcp.set_title(f"TCP (Spitze / {TCP_BODY}) x-z")
        self.ax_tcp.set_xlabel("x (m)")
        self.ax_tcp.set_ylabel("z (m)")
        self.ax_tcp.grid(True, ls="--", alpha=0.4)
        self.ax_tcp.set_aspect("equal", adjustable="datalim")

        self.M_img = self.ax_M.imshow(
            np.zeros((self.nv, self.nv)), cmap="viridis", aspect="auto", origin="upper"
        )
        self.ax_M.set_title("Massenmatrix M(q)  (nv×nv)")
        self.M_cbar = self.fig.colorbar(self.M_img, ax=self.ax_M, fraction=0.046, pad=0.04)

        idx = np.arange(self.nv)
        self.bias_bars = self.ax_bias.bar(idx, np.zeros(self.nv), color="darkorange")
        self.ax_bias.set_title("Bias force c(q,v)  (qfrc_bias)")
        self.ax_bias.set_xlabel("dof")
        self.ax_bias.grid(True, ls="--", alpha=0.3, axis="y")

        self.tau_bars = self.ax_tau.bar(idx, np.zeros(self.nv), color="seagreen")
        self.ax_tau.set_title("Applied force tau  (passive+act+applied)")
        self.ax_tau.set_xlabel("dof")
        self.ax_tau.grid(True, ls="--", alpha=0.3, axis="y")

        self.ax_efc.set_title("Constraint forces f  (efc_force)")
        self.ax_efc.set_xlabel("constraint #")

        self.fig.canvas.draw()
        plt.show(block=False)

    def clear_history(self) -> None:
        self.t_hist.clear(); self.q_hist.clear(); self.v_hist.clear(); self.tcp_hist.clear()

    def _trim_window(self) -> None:
        if not self.t_hist:
            return
        t_now = self.t_hist[-1]
        while self.t_hist and (t_now - self.t_hist[0]) > self.window_s:
            self.t_hist.popleft(); self.q_hist.popleft(); self.v_hist.popleft()

    def update(self, snap: dict) -> None:
        self._frame += 1

        # ── realtime factor in the title ──
        rtf = snap["rtf"]
        self.suptitle.set_text(
            f"Realtime factor: {rtf:.2f}×     |     sim t = {snap['t']:.1f} s     "
            f"|     nq={self.nv}  nv={self.nv}  nefc={snap['nefc']}"
        )
        self.suptitle.set_color("green" if rtf >= 0.95 else ("darkorange" if rtf >= 0.5 else "red"))

        # ── record time-series ──
        self.t_hist.append(snap["t"])
        self.q_hist.append(snap["q"])
        self.v_hist.append(snap["v"])
        self._trim_window()
        self.tcp_hist.append((float(snap["tcp"][0]), float(snap["tcp"][2])))

        t = np.asarray(self.t_hist)
        q = np.asarray(self.q_hist)
        v = np.asarray(self.v_hist)
        nq_plot = min(q.shape[1], self.nv)
        for i in range(nq_plot):
            self.q_lines[i].set_data(t, q[:, i])
            self.v_lines[i].set_data(t, v[:, i])
        for ax in (self.ax_q, self.ax_v):
            ax.relim(); ax.autoscale_view()

        # ── TCP trajectory ──
        tx = [p[0] for p in self.tcp_hist]
        tz = [p[1] for p in self.tcp_hist]
        self.tcp_trace.set_data(tx, tz)
        self.tcp_dot.set_data([tx[-1]], [tz[-1]])
        self.ax_tcp.relim(); self.ax_tcp.autoscale_view()

        # ── mass matrix (decimated: expensive imshow update) ──
        if self._frame % self.matrix_every == 0:
            M = snap["M"]
            self.M_img.set_data(M)
            lo, hi = float(M.min()), float(M.max())
            self.M_img.set_clim(lo, hi if hi > lo else lo + 1e-9)

        # ── bias + applied force bars ──
        c, tau = snap["bias"], snap["tau"]
        for bar, val in zip(self.bias_bars, c):
            bar.set_height(float(val))
        for bar, val in zip(self.tau_bars, tau):
            bar.set_height(float(val))
        self.ax_bias.set_ylim(*_sym_range(c))
        self.ax_tau.set_ylim(*_sym_range(tau))

        # ── constraint forces (variable length → redraw) ──
        self.ax_efc.clear()
        nefc = snap["nefc"]
        if nefc > 0:
            self.ax_efc.bar(np.arange(nefc), snap["efc"], color="slateblue")
        else:
            self.ax_efc.text(0.5, 0.5, "no active\nconstraints", ha="center", va="center",
                             transform=self.ax_efc.transAxes, color="gray")
        self.ax_efc.set_title(f"Constraint forces f  (nefc={nefc})")
        self.ax_efc.set_xlabel("constraint #")
        self.ax_efc.grid(True, ls="--", alpha=0.3, axis="y")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)


def _sym_range(vals: np.ndarray) -> tuple[float, float]:
    m = float(np.max(np.abs(vals))) if len(vals) else 1.0
    m = max(m, 1e-6)
    return -1.1 * m, 1.1 * m


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=str, default=str(DEFAULT_XML), help="Model XML path")
    ap.add_argument("--params", type=str, default=str(DEFAULT_PARAMS),
                    help="Identified-parameter JSON (sysid_real_modular_*.json)")
    ap.add_argument("--no-params", action="store_true", help="Do NOT apply identified params (use raw XML)")
    ap.add_argument("--window", type=float, default=8.0, help="Rolling time-plot window (s)")
    ap.add_argument("--plot-hz", type=float, default=20.0, help="Dashboard refresh rate (Hz)")
    ap.add_argument("--matrix-every", type=int, default=2,
                    help="Update the mass-matrix heatmap every N plot frames (>1 = cheaper draws)")
    ap.add_argument("--init-ctrl", type=float, default=0.0,
                    help="Initial tendon force setpoint applied to all actuators (N, positive=tension)")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path
    model = mj.MjModel.from_xml_path(str(xml_path))

    # match the sys-id integration settings so the dynamics agree with the fit
    model.opt.timestep = 0.004
    model.opt.iterations = 20

    if not args.no_params:
        params_path = Path(args.params)
        if not params_path.is_absolute():
            params_path = REPO_ROOT / params_path
        if params_path.exists():
            print(f"Applied identified parameters: {apply_identified_params(model, params_path)}")
        else:
            print(f"[warn] params file not found ({params_path}) — using raw XML values")
    else:
        print("Using raw XML parameters (--no-params).")

    data = mj.MjData(model)
    for i in range(model.nu):
        data.ctrl[i] = -abs(args.init_ctrl)   # ctrlrange [-50,0], negative = tension
    mj.mj_forward(model, data)

    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, ntendon={model.ntendon}")
    print("MuJoCo viewer: drag the Control-panel actuator sliders to steer. Press 'R' to reset.")
    print("A separate Matplotlib window shows the live dynamics dashboard (realtime factor in the title).")

    # Give the physics thread frequent GIL hand-offs so it keeps up during draws.
    sys.setswitchinterval(0.0005)

    shared = SharedState(model, data)
    dash = LiveDashboard(model, window_s=args.window, matrix_every=args.matrix_every)

    phys = threading.Thread(target=physics_thread, args=(shared, args.init_ctrl), daemon=True)
    phys.start()

    plot_interval = 1.0 / max(args.plot_hz, 1e-3)
    try:
        while not shared.stop.is_set() and dash.is_open():
            frame_start = time.perf_counter()
            if shared.reset_flag.is_set():
                dash.clear_history()
                shared.reset_flag.clear()
            dash.update(shared.snapshot())
            sleep = plot_interval - (time.perf_counter() - frame_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        shared.stop.set()
        phys.join(timeout=2.0)
        plt.close("all")
    print("Closed — exiting.")


if __name__ == "__main__":
    main()
