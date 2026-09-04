"""Multi-parameter SpiRob system identification in a sim-to-sim scenario.

A ground-truth model with known (slightly scattered) parameters is simulated;
the same parameters are then identified back from that reference trajectory.
Because the true values are known, the identification error can be stated
directly -- this is the best case, an upper bound on achievable quality, since
there is no friction, no measurement noise and no model mismatch.

Two optimizers::

    --optimizer de     Differential Evolution (scipy, default)
    --optimizer cma    CMA-ES (package 'cma'), searching a normalized [0,1] box

Usage::

    uv run sysid/simulation_based/sim2sim.py --sim-time 4 --maxiter 200 --init-angle 0
    uv run sysid/simulation_based/sim2sim.py --optimizer cma --sim-time 4 --maxiter 200
    uv run sysid/simulation_based/sim2sim.py --compare          # DE vs CMA, all configs
    uv run sysid/simulation_based/sim2sim.py --compare --configs "4:200:0,20:200:0"
    uv run sysid/simulation_based/sim2sim.py --sensitivity      # identifiability study

In comparison mode (``--compare``) DE runs first with the given number of
generations; CMA then receives the *same budget of function evaluations*
(``--budget-mode evals``, the default) so both methods get equal compute. With
``--budget-mode generations`` both get the same generation count, which is NOT
budget-fair: a DE generation is far larger than a CMA generation.

Outputs: build/sim2sim/*.{json,png}
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from scipy.optimize import differential_evolution

import spirob.generator as sg
from spirob.paths import build_dir as _build_dir

# ── Modell-Geometrie (feststehend) ──────────────────────────────────
L_TARGET = 0.44
BASE_D = 0.1
TIP_D = 0.03
DELTA_THETA_DEG = 30.0

# ── Ground-Truth-Base-Parameter ────────────────────────────────────
GT_BASE_STIFFNESS = 0.08
GT_BASE_DAMPING = 0.12
GT_BASE_TENDON_STIFFNESS = 500.0
# GT_BASE_ARMATURE = 0.015

# ── Initial values for the optimization (init base) ────────────────
INIT_BASE_STIFFNESS = 0.8
INIT_BASE_DAMPING = 0.02
INIT_BASE_TENDON_STIFFNESS = 500.0
# INIT_BASE_ARMATURE = 0.005

# ── Suchraum-Grenzen (physikalisch sinnvoll) ──────────────────────
BOUNDS_STIFFNESS = (0.01, 1)
BOUNDS_DAMPING = (0.01, 0.2)
BOUNDS_TENDON_STIFFNESS = (450.0, 550.0)
# BOUNDS_ARMATURE = (1e-3, 0.1)

# ── Initial joint configuration (start pose) ───────────────────────
# This angle (in degrees) is applied to every joint.
# Default for single runs; in comparison mode it comes from --configs.
INIT_JOINT_ANGLE_DEG = 20.0  # 0 = zero pose; e.g. 20.0 puts 20 deg on every joint

# ── Configurations of the comparison run ───────────────────────────
# (recording duration [s], iterations/generations, start pose [deg])
# These correspond to the rows of the result table in the thesis.
DEFAULT_COMPARE_CONFIGS: list[tuple[float, int, float]] = [
    (4.0, 200, 0.0),
    (4.0, 400, 0.0),
    (20.0, 200, 0.0),
    (20.0, 400, 0.0),
    (10.0, 400, 20.0),
]

BUILD_DIR = _build_dir("sim2sim")


# ====================================================================
#  Hilfsfunktionen
# ====================================================================

def make_model() -> mj.MjModel:
    xml = sg.generate_xml_string(
        L_target=L_TARGET,
        base_d=BASE_D,
        tip_d=TIP_D,
        Delta_theta_deg=DELTA_THETA_DEG,
        auto_format=True,
    )
    return mj.MjModel.from_xml_string(xml)

def set_params(model: mj.MjModel,
               stiffness: np.ndarray,
               damping: np.ndarray,
               tendon_stiffness: np.ndarray) -> None:
    """Write per-element physical parameters (arrays) to every joint / tendon."""
    for i in range(model.njnt):
        model.jnt_stiffness[i] = stiffness[i]
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof] = damping[i]
        # model.dof_armature[dof] = armature[i]
    for i in range(model.ntendon):
        model.tendon_stiffness[i] = tendon_stiffness[i]

def controller(model: mj.MjModel, data: mj.MjData, t: float) -> None:
    # ramp = min(t / 0.5, 1.0)
    # s1 = np.sin(2 * np.pi * 0.5 * t)
    # s2 = np.sin(2 * np.pi * 1.5 * t)
    # s3 = np.sin(2 * np.pi * 3.0 * t)
    # data.ctrl[0] = -ramp * (5.0 + 3.0 * s1 + 1.5 * s2)
    # data.ctrl[1] = -ramp * (8.0 + 4.0 * s1 - 2.0 * s3)
    data.ctrl[0] = -30.0 * np.sin(2 * np.pi * 0.5 * t)
    data.ctrl[1] = -20.0 * np.sin(2 * np.pi * 0.5 * t + np.pi / 4)

def simulate(model: mj.MjModel, sim_time: float,
             record_dt: float = 0.1,
             init_angle_deg: float = INIT_JOINT_ANGLE_DEG) -> tuple[np.ndarray, np.ndarray]:
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Set the initial joint positions (in radians)
    init_angle_rad = np.deg2rad(init_angle_deg)
    for i in range(model.njnt):
        data.qpos[i] = init_angle_rad
    mj.mj_forward(model, data)  # Update internal state

    model.opt.timestep = 0.02
    dt = model.opt.timestep
    record_every = max(1, round(record_dt / dt))
    total_steps = int(sim_time / dt)

    qpos_list, qvel_list = [], []
    for step in range(total_steps + 1):
        if step % record_every == 0:
            qpos_list.append(data.qpos.copy())
            qvel_list.append(data.qvel.copy())
        controller(model, data, data.time)
        mj.mj_step(model, data)
    return np.array(qpos_list), np.array(qvel_list)

def decode_params(x_scaled: np.ndarray, njnt: int, ntendon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the three parameter arrays from the flat scalar vector."""
    stiffness = x_scaled[0:njnt]
    damping = x_scaled[njnt:2*njnt]
    # armature = x_scaled[2*njnt:3*njnt]
    tendon_stiffness = x_scaled[2*njnt:2*njnt+ntendon]
    return stiffness, damping, tendon_stiffness

# ====================================================================
#  Multiprocessing-Hilfen
# ====================================================================

_LOCAL_MODEL = None

def get_local_model() -> mj.MjModel:
    """Cache the MuJoCo model per thread/process -- MjModel is not picklable."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        _LOCAL_MODEL = make_model()
    return _LOCAL_MODEL

def global_objective(x: np.ndarray, scale: np.ndarray, gt_qpos: np.ndarray, gt_qvel: np.ndarray,
                     sim_time: float, njnt: int, ntendon: int, init_angle_deg: float) -> float:
    """Module-level objective, so it survives pickling under multiprocessing (workers>1)."""
    model = get_local_model()
    # Timing is disabled under MP to avoid race conditions and sync overhead
    return cost_function(x, scale, gt_qpos, gt_qvel, sim_time, model, njnt, ntendon,
                         init_angle_deg, timing=None)


# CMA-ES evaluates through its own pool; the context is built BEFORE the fork
# set before the fork so the child processes inherit it (Linux fork, as for DE).
_CMA_CTX: tuple | None = None


def _pool_init_ignore_sigint() -> None:
    """Workers ignore SIGINT so Ctrl+C only reaches the parent process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _cma_eval(x_phys: np.ndarray) -> float:
    assert _CMA_CTX is not None, "_CMA_CTX was not set before the fork"
    return global_objective(x_phys, *_CMA_CTX)

# ====================================================================
#  Kostenfunktion
# ====================================================================

def cost_function(params_norm: np.ndarray,
                  scale: np.ndarray,
                  gt_qpos: np.ndarray,
                  gt_qvel: np.ndarray,
                  sim_time: float,
                  model: mj.MjModel,
                  njnt: int,
                  ntendon: int,
                  init_angle_deg: float = INIT_JOINT_ANGLE_DEG,
                  timing: dict[str, Any] | None = None) -> float:
    t_cost_start = time.perf_counter()
    phys = params_norm * scale

    if timing is not None:
        timing["cost_calls"] += 1

    if np.any(phys <= 0):
        if timing is not None:
            timing["cost_invalid_param_calls"] += 1
            timing["cost_total_s"] += time.perf_counter() - t_cost_start
        return 1e6

    stiff, damp, t_stiff = decode_params(phys, njnt, ntendon)

    try:
        t0 = time.perf_counter()
        set_params(model, stiff, damp, t_stiff)
        if timing is not None:
            timing["cost_set_params_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        est_qpos, est_qvel = simulate(model, sim_time, init_angle_deg=init_angle_deg)
        if timing is not None:
            timing["cost_simulate_s"] += time.perf_counter() - t0
    except Exception:
        if timing is not None:
            timing["cost_exception_calls"] += 1
            timing["cost_total_s"] += time.perf_counter() - t_cost_start
        return 1e6

    t0 = time.perf_counter()
    n = min(len(gt_qpos), len(est_qpos))
    w = np.linspace(0.5, 1.5, n).reshape(-1, 1)
    mse_pos = np.mean(w * (gt_qpos[:n] - est_qpos[:n]) ** 2)
    mse_vel = np.mean(w * (gt_qvel[:n] - est_qvel[:n]) ** 2)

    # Instability guard: a diverging simulation returns NaN/Inf; a finite
    # penalty keeps the optimizer working instead of poisoning it.
    if not np.isfinite(mse_pos) or not np.isfinite(mse_vel):
        return 1e6

    if timing is not None:
        timing["cost_postprocess_s"] += time.perf_counter() - t0
        timing["cost_total_s"] += time.perf_counter() - t_cost_start

    return mse_pos + 0.5 * mse_vel


# ====================================================================
#  Ground Truth & Suchraum
# ====================================================================

@dataclass
class GroundTruth:
    njnt: int
    ntendon: int
    stiffness: np.ndarray
    damping: np.ndarray
    tendon_stiffness: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


def build_ground_truth(sim_time: float, init_angle_deg: float, seed: int = 42,
                       verbose: bool = True) -> GroundTruth:
    """Build the ground-truth model with scattered parameters and its reference trajectory."""
    gt_model = make_model()
    njnt = gt_model.njnt
    ntendon = gt_model.ntendon

    # -------------------------------------------------------------
    # MANUAL GROUND-TRUTH VALUES HERE (arrays of the matching length)
    # Slightly scattered values around the base parameter, as an example
    # (so each joint responds slightly differently; edit by hand if needed)
    # -------------------------------------------------------------
    # Legacy-RNG bewusst beibehalten: erzeugt exakt dieselbe Ground Truth wie
    # earlier runs, so old and new results stay comparable.
    np.random.seed(seed)
    gt_stiffness = np.random.normal(GT_BASE_STIFFNESS, 0.005, njnt)
    gt_damping = np.random.normal(GT_BASE_DAMPING, 0.005, njnt)
    # gt_armature = np.random.normal(GT_BASE_ARMATURE, 0.001, njnt)
    gt_tendon_stiffness = np.random.normal(GT_BASE_TENDON_STIFFNESS, 4.0, ntendon)

    # To pin fixed arrays, simply overwrite them here, e.g.:
    # gt_stiffness = np.array([0.08, 0.09, 0.08, 0.1, ... <njnt values>])

    if verbose:
        print(f"  Modell: {njnt} Joints, {ntendon} Tendons")
        print(f"  Suchraum: {2 * njnt + ntendon} Dimensionen (Parameter) total!")
        print("\n  --- Ground-truth values ---")
        print(f"  GT Stiffness: {np.array2string(gt_stiffness, precision=4, max_line_width=120)}")
        print(f"  GT Damping:   {np.array2string(gt_damping, precision=4, max_line_width=120)}")
        print(f"  GT T_Stiff:   {np.array2string(gt_tendon_stiffness, precision=2, max_line_width=120)}")
        print("  --------------------------\n")

    set_params(gt_model, gt_stiffness, gt_damping, gt_tendon_stiffness)
    gt_qpos, gt_qvel = simulate(gt_model, sim_time, init_angle_deg=init_angle_deg)
    if verbose:
        print(f"  {len(gt_qpos)} Datenpunkte simuliert")

    return GroundTruth(njnt, ntendon, gt_stiffness, gt_damping, gt_tendon_stiffness,
                       gt_qpos, gt_qvel)


def build_search_space(njnt: int, ntendon: int) -> tuple[np.ndarray, list[tuple[float, float]], np.ndarray]:
    """Scaling, normalized bounds and start vector (shared by all optimizers)."""
    scale = np.concatenate([
        np.full(njnt, INIT_BASE_STIFFNESS),
        np.full(njnt, INIT_BASE_DAMPING),
        # np.full(njnt, INIT_BASE_ARMATURE),
        np.full(ntendon, INIT_BASE_TENDON_STIFFNESS)
    ])

    norm_bounds: list[tuple[float, float]] = []
    for _ in range(njnt):
        norm_bounds.append((BOUNDS_STIFFNESS[0]/INIT_BASE_STIFFNESS, BOUNDS_STIFFNESS[1]/INIT_BASE_STIFFNESS))
    for _ in range(njnt):
        norm_bounds.append((BOUNDS_DAMPING[0]/INIT_BASE_DAMPING, BOUNDS_DAMPING[1]/INIT_BASE_DAMPING))
    # for _ in range(njnt): norm_bounds.append((BOUNDS_ARMATURE[0]/INIT_BASE_ARMATURE, BOUNDS_ARMATURE[1]/INIT_BASE_ARMATURE))
    for _ in range(ntendon):
        norm_bounds.append((BOUNDS_TENDON_STIFFNESS[0]/INIT_BASE_TENDON_STIFFNESS,
                            BOUNDS_TENDON_STIFFNESS[1]/INIT_BASE_TENDON_STIFFNESS))

    x0 = np.ones(len(scale))
    return scale, norm_bounds, x0


# ====================================================================
#  Optimizer backends  (identical search space, identical cost function)
# ====================================================================

@dataclass
class OptResult:
    optimizer: str
    x_norm: np.ndarray
    cost: float
    nfev: int
    elapsed_s: float
    generations: int
    cost_history: list[float]
    message: str = ""


def run_de(ctx: tuple, norm_bounds, x0, maxiter: int, tol: float, workers: int,
           verbose: bool = True) -> OptResult:
    """Differential Evolution (scipy). popsize is per dimension."""
    cost_history: list[float] = []
    gen = 0

    def callback_collect_cost(xk, convergence=None):
        nonlocal gen
        gen += 1
        cost_history.append(global_objective(xk, *ctx))
        if verbose:
            print(f"[de gen {gen:04d}/{maxiter}] best_cost={cost_history[-1]:.6e}")

    popsize = 2 * max(1, workers)
    if verbose:
        print(f"[de setup] popsize={popsize}/dim, total_population={popsize * len(norm_bounds)}, "
              f"maxiter={maxiter}, workers={workers}")

    t0 = time.time()
    result = differential_evolution(
        global_objective,
        args=ctx,
        bounds=norm_bounds,
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        seed=42,
        polish=True,
        init="sobol",
        popsize=popsize,
        recombination=0.9,
        workers=workers,
        disp=False,
        callback=callback_collect_cost,
    )
    elapsed = time.time() - t0
    # Callback evaluations are extra simulations and count against the budget.
    nfev = int(result.nfev) + len(cost_history)
    return OptResult("de", np.asarray(result.x), float(result.fun), nfev, elapsed,
                     gen, cost_history, str(result.message))


def cma_popsize(ndim: int, workers: int) -> int:
    """CMA-ES default population size, at least as large as the worker count."""
    return max(4 + int(3 * np.log(ndim)), int(max(1, workers)))


def run_cma(ctx: tuple, norm_bounds, x0, maxiter: int, workers: int, sigma0: float = 0.25,
            verbose: bool = True) -> OptResult:
    """CMA-ES in a normalized [0,1] box.

    The box normalisation matters because the parameter groups span very
    span very different magnitudes (stiffness ~0.08, damping ~0.12,
    tendon stiffness ~500) and CMA-ES works from a single initial step size
    sigma0 startet.
    """
    global _CMA_CTX
    try:
        import cma
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CMA-ES needs the 'cma' package — install with: uv add cma") from exc

    lo = np.array([b[0] for b in norm_bounds], dtype=float)
    hi = np.array([b[1] for b in norm_bounds], dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)
    x0n = np.clip((np.asarray(x0, dtype=float) - lo) / span, 0.0, 1.0)

    def denorm(xn):
        return lo + np.asarray(xn, dtype=float) * span

    popsize = cma_popsize(len(norm_bounds), workers)
    es = cma.CMAEvolutionStrategy(
        list(x0n), sigma0,
        {"bounds": [0, 1], "maxiter": maxiter, "popsize": popsize, "seed": 42, "verbose": -9},
    )
    if verbose:
        print(f"[cma setup] popsize={popsize}, sigma0={sigma0}, maxiter={maxiter}, workers={workers}")

    pool = None
    if workers and workers > 1:
        import multiprocessing as mp
        _CMA_CTX = ctx  # set BEFORE the fork so the children inherit it
        pool = mp.Pool(processes=workers, initializer=_pool_init_ignore_sigint)

    cost_history: list[float] = []
    gen = 0
    nfev = 0
    t0 = time.time()
    try:
        while not es.stop():
            gen += 1
            sols = es.ask()
            phys = [denorm(s) for s in sols]
            if pool is not None:
                costs = pool.map(_cma_eval, phys)
            else:
                costs = [global_objective(p, *ctx) for p in phys]
            es.tell(sols, costs)
            nfev += len(sols)
            cost_history.append(float(es.result.fbest))
            if verbose:
                print(f"[cma gen {gen:04d}/{maxiter}] best_cost={cost_history[-1]:.6e} sigma={es.sigma:.3e}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    elapsed = time.time() - t0

    x_best = denorm(es.result.xbest)
    return OptResult("cma", np.asarray(x_best), float(es.result.fbest), nfev, elapsed,
                     gen, cost_history, str(es.stop()))


# ====================================================================
#  Error measures
# ====================================================================

def parameter_errors(gt: GroundTruth, stiff: np.ndarray, damp: np.ndarray,
                     t_stiff: np.ndarray) -> dict[str, float]:
    """Relative errors, in percent.

    ``*_mean_err``  : error of the group means (|mean(id)-mean(gt)|/mean(gt)) --
                      the measure used originally. Careful: errors of
                      individual joints can cancel each other here.
    ``*_mape``      : mean absolute percentage error over all elements —
                      the stricter measure.
    """
    def mean_err(gt_v: np.ndarray, id_v: np.ndarray) -> float:
        g = float(np.mean(gt_v))
        return abs(float(np.mean(id_v)) - g) / g * 100.0 if g else 0.0

    def mape(gt_v: np.ndarray, id_v: np.ndarray) -> float:
        return float(np.mean(np.abs(id_v - gt_v) / np.abs(gt_v)) * 100.0)

    return {
        "stiffness_mean_err": mean_err(gt.stiffness, stiff),
        "damping_mean_err": mean_err(gt.damping, damp),
        "tendon_mean_err": mean_err(gt.tendon_stiffness, t_stiff),
        "stiffness_mape": mape(gt.stiffness, stiff),
        "damping_mape": mape(gt.damping, damp),
        "tendon_mape": mape(gt.tendon_stiffness, t_stiff),
    }


def print_result_tables(gt: GroundTruth, stiff: np.ndarray, damp: np.ndarray,
                        t_stiff: np.ndarray, errs: dict[str, float]) -> None:
    print("  Mean parameter errors:")
    print(f"  {'Parameter':>20s}  {'GT Mean':>10s}  {'Identif.':>10s}  {'Error':>8s}  {'MAPE':>8s}")
    print("  " + "-" * 64)
    rows = [
        ("joint_stiffness", gt.stiffness, stiff, "stiffness"),
        ("joint_damping", gt.damping, damp, "damping"),
        ("tendon_stiffness", gt.tendon_stiffness, t_stiff, "tendon"),
    ]
    for lbl, gt_v, id_v, key in rows:
        print(f"  {lbl:>20s}  {np.mean(gt_v):10.5f}  {np.mean(id_v):10.5f}  "
              f"{errs[key + '_mean_err']:7.2f} %  {errs[key + '_mape']:7.2f} %")

    print("\n  Detailed parameter errors (PER JOINT / TENDON):")
    print(f"  {'Index':>5s} | {'Stiffness (GT/Id/Err%)':>25s} | {'Damping (GT/Id/Err%)':>25s}")
    print("  " + "-" * 60)
    for i in range(gt.njnt):
        err_s = abs(stiff[i] - gt.stiffness[i]) / gt.stiffness[i] * 100 if gt.stiffness[i] else 0
        err_d = abs(damp[i] - gt.damping[i]) / gt.damping[i] * 100 if gt.damping[i] else 0
        s_str = f"{gt.stiffness[i]:.4f}/{stiff[i]:.4f}/{err_s:5.1f}%"
        d_str = f"{gt.damping[i]:.4f}/{damp[i]:.4f}/{err_d:5.1f}%"
        print(f"  {i:5d} | {s_str:>25s} | {d_str:>25s}")

    if gt.ntendon > 0:
        print(f"\n  {'Index':>5s} | {'Tendon Stiff (GT/Id/Err%)':>25s}")
        print("  " + "-" * 35)
        for i in range(gt.ntendon):
            err_ts = (abs(t_stiff[i] - gt.tendon_stiffness[i]) / gt.tendon_stiffness[i] * 100
                      if gt.tendon_stiffness[i] else 0)
            print(f"  {i:5d} | {f'{gt.tendon_stiffness[i]:.2f}/{t_stiff[i]:.2f}/{err_ts:5.1f}%':>25s}")
    print("=" * 55)


# ====================================================================
#  One experiment (one configuration, one optimizer)
# ====================================================================

def run_experiment(sim_time: float, maxiter: int, init_angle_deg: float, optimizer: str,
                   workers: int, tol: float, sigma0: float, gt: GroundTruth | None = None,
                   verbose: bool = True, seed: int = 42) -> dict[str, Any]:
    """Run one identification and return error, wall time and budget."""
    if gt is None:
        gt = build_ground_truth(sim_time, init_angle_deg, seed=seed, verbose=verbose)

    scale, norm_bounds, x0 = build_search_space(gt.njnt, gt.ntendon)
    ctx = (scale, gt.qpos, gt.qvel, sim_time, gt.njnt, gt.ntendon, init_angle_deg)

    if optimizer == "de":
        res = run_de(ctx, norm_bounds, x0, maxiter, tol, workers, verbose=verbose)
    elif optimizer == "cma":
        res = run_cma(ctx, norm_bounds, x0, maxiter, workers, sigma0=sigma0, verbose=verbose)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}' (de | cma)")

    phys = res.x_norm * scale
    stiff, damp, t_stiff = decode_params(phys, gt.njnt, gt.ntendon)
    errs = parameter_errors(gt, stiff, damp, t_stiff)

    if verbose:
        print("\n" + "=" * 55 + f"\nRESULT ({optimizer.upper()})")
        print(f"  Status  : {res.message}\n  Cost    : {res.cost:.10e}\n"
              f"  nfev    : {res.nfev}\n  Gen.    : {res.generations}\n  Time    : {res.elapsed_s:.1f} s\n")
        print_result_tables(gt, stiff, damp, t_stiff, errs)

    return {
        "optimizer": optimizer,
        "sim_time": sim_time,
        "maxiter": maxiter,
        "init_angle_deg": init_angle_deg,
        "generations": res.generations,
        "nfev": res.nfev,
        "elapsed_s": res.elapsed_s,
        "cost": res.cost,
        **errs,
        "identified": {
            "stiffness": stiff.tolist(),
            "damping": damp.tolist(),
            "tendon_stiffness": t_stiff.tolist(),
        },
        "ground_truth": {
            "stiffness": gt.stiffness.tolist(),
            "damping": gt.damping.tolist(),
            "tendon_stiffness": gt.tendon_stiffness.tolist(),
        },
        "cost_history": [float(c) for c in res.cost_history],
    }


# ====================================================================
#  Vergleichsmodus DE vs. CMA
# ====================================================================

def parse_configs(spec: str | None) -> list[tuple[float, int, float]]:
    """'4:200:0,20:400:0' → [(4.0, 200, 0.0), (20.0, 400, 0.0)]"""
    if not spec:
        return list(DEFAULT_COMPARE_CONFIGS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--configs: erwartet sim_time:maxiter:startwinkel, bekam '{tok}'")
        out.append((float(parts[0]), int(parts[1]), float(parts[2])))
    return out


def run_comparison(configs: list[tuple[float, int, float]], workers: int, tol: float,
                   sigma0: float, budget_mode: str, seed: int) -> list[dict[str, Any]]:
    """Run DE and CMA on identical ground truth for every configuration."""
    records: list[dict[str, Any]] = []
    for k, (sim_time, maxiter, angle) in enumerate(configs, 1):
        print("\n" + "#" * 70)
        print(f"# Konfiguration {k}/{len(configs)}: sim_time={sim_time}s, maxiter={maxiter}, "
              f"start pose={angle} deg")
        print("#" * 70)

        # The same ground truth for both optimizers -- otherwise the comparison is worthless.
        gt = build_ground_truth(sim_time, angle, seed=seed, verbose=True)

        print("\n--- Differential Evolution ---")
        rec_de = run_experiment(sim_time, maxiter, angle, "de", workers, tol, sigma0,
                                gt=gt, verbose=True, seed=seed)
        records.append(rec_de)

        if budget_mode == "evals":
            # CMA bekommt dasselbe Budget an Funktionsauswertungen wie DE verbraucht hat.
            ndim = 2 * gt.njnt + gt.ntendon
            ps = cma_popsize(ndim, workers)
            cma_iter = max(1, int(np.ceil(rec_de["nfev"] / ps)))
            print(f"\n[budget] DE nfev={rec_de['nfev']} → CMA maxiter={cma_iter} "
                  f"(popsize={ps}) for an equal evaluation budget")
        else:
            cma_iter = maxiter
            print(f"\n[budget] same generation count for both: {cma_iter}")

        print("\n--- CMA-ES ---")
        rec_cma = run_experiment(sim_time, cma_iter, angle, "cma", workers, tol, sigma0,
                                 gt=gt, verbose=True, seed=seed)
        rec_cma["maxiter"] = maxiter  # zugeordnete Tabellenzeile
        rec_cma["cma_maxiter"] = cma_iter
        records.append(rec_cma)

    return records


def print_comparison_summary(records: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("COMPARISON DE vs CMA-ES")
    print("=" * 100)
    hdr = (f"{'Opt.':>6} {'simT[s]':>9} {'Iter':>6} {'Start':>7} {'nfev':>8} {'Wall[s]':>9} "
           f"{'Cost':>12} {'ErrK%':>7} {'ErrD%':>7} {'ErrS%':>7} {'MAPE_K%':>8} {'MAPE_D%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        print(f"{r['optimizer'].upper():>6} {r['sim_time']:>9.0f} {r['maxiter']:>6d} "
              f"{r['init_angle_deg']:>6.0f}° {r['nfev']:>8d} {r['elapsed_s']:>9.1f} "
              f"{r['cost']:>12.4e} {r['stiffness_mean_err']:>7.1f} {r['damping_mean_err']:>7.1f} "
              f"{r['tendon_mean_err']:>7.1f} {r['stiffness_mape']:>8.1f} {r['damping_mape']:>8.1f}")
    print("=" * 100)


def _fmt_de(x: float, digits: int = 1) -> str:
    """Number with a German decimal comma (this table is for the thesis)."""
    return f"{x:.{digits}f}".replace(".", ",")


def typst_table(records: list[dict[str, Any]]) -> str:
    """Result table in Typst format, in **German** -- this feeds the thesis, not the docs."""
    lines = [
        "#figure(",
        "  table(",
        "    columns: (auto, 2cm, 2cm, auto, auto, auto, auto, auto),",
        "    align: (left, right, right, right, right, right, right, right),",
        "    stroke: none,",
        "    table.hline(y: 0, stroke: 1pt),",
        "    table.hline(y: 1, stroke: 0.5pt),",
        f"    table.hline(y: {len(records) + 1}, stroke: 1pt),",
        "    [Verfahren], [Aufnahme- \\ dauer], [Iterationen], [Startlage], "
        "[Time \\ [s]], [Error \\ stiffness], [Error \\ damping], [Error \\ tendon],",
    ]
    for r in records:
        name = "DE" if r["optimizer"] == "de" else "CMA-ES"
        pose = "0° (Ruhelage)" if r["init_angle_deg"] == 0 else f"{r['init_angle_deg']:.0f}° (Ausgelenkt)"
        lines.append(
            f"    [{name}], [{r['sim_time']:.0f} s], [{r['maxiter']}], [{pose}], "
            f"[{_fmt_de(r['elapsed_s'])}], [{_fmt_de(r['stiffness_mean_err'])} %], "
            f"[{_fmt_de(r['damping_mean_err'])} %], [{_fmt_de(r['tendon_mean_err'])} %],"
        )
    lines += [
        "  ),",
        "  caption: [Vergleich der mittleren Parameterfehler von Differential Evolution und "
        "CMA-ES bei unterschiedlichen Optimierungskonfigurationen im Sim-to-Sim-Szenario. "
        "Beide Verfahren arbeiten auf derselben Ground Truth und mit demselben Budget an "
        "Funktionsauswertungen.],",
        ") <tab:sysid_results>",
    ]
    return "\n".join(lines)


def plot_comparison(records: list[dict[str, Any]], path: Path) -> None:
    """Convergence traces, DE vs CMA, one subplot per configuration."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [Warn] comparison plot not possible: {exc}")
        return

    keys = []
    for r in records:
        k = (r["sim_time"], r["maxiter"], r["init_angle_deg"])
        if k not in keys:
            keys.append(k)
    n = len(keys)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)

    for idx, k in enumerate(keys):
        ax = axs[idx // ncols][idx % ncols]
        for r in records:
            if (r["sim_time"], r["maxiter"], r["init_angle_deg"]) != k:
                continue
            hist = r["cost_history"]
            if not hist:
                continue
            # x-Achse: kumulierte Funktionsauswertungen → budgetfairer Vergleich
            x = np.linspace(0, r["nfev"], len(hist))
            ax.semilogy(x, hist, marker="o", markersize=2.5, linewidth=1.5,
                        label=f"{r['optimizer'].upper()} (Endkosten {r['cost']:.2e})")
        ax.set_title(f"{k[0]:.0f} s, {k[1]} Iter., Start {k[2]:.0f}°")
        ax.set_xlabel("Funktionsauswertungen")
        ax.set_ylabel("Bester Kostenwert")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Info] comparison plot saved to: {path}")


# ====================================================================
#  Sensitivity / identifiability analysis
# ====================================================================
#
# At the ground-truth point the cost function is exactly zero and its gradient
# vanishes (global minimum), so a first-order sensitivity says nothing there.
# Two measures are computed instead:
#
#   1) Cost rise under a relative parameter deviation (OAT scan) -- how hard
#      does the cost function punish a mis-estimate of the group?
#   2) Trajectory sensitivity S = RMS(dq) in degrees for a +10 % change of a
#      single parameter -- how much does the parameter move the observable at
#      all? If S sits below the measurement noise, the parameter is simply not
#      identifiable from the data.
#
# The collinearity of the groups is computed as well: the cosine between the
# trajectory changes of two groups. Values near +/-1 mean the groups can
# compensate for each other, so the parameters are identifiable only as a
# combination, never individually.

_SENS_CTX: dict | None = None

SENS_GROUPS = ("stiffness", "damping", "tendon_stiffness")


def _sens_simulate(phys: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from physical parameter arrays (safe to call in a worker)."""
    assert _SENS_CTX is not None, "_SENS_CTX was not set before the fork"
    model = get_local_model()
    set_params(model, *phys)
    return simulate(model, _SENS_CTX["sim_time"], init_angle_deg=_SENS_CTX["init_angle_deg"])


def _sens_cost(phys) -> float:
    """Optimization cost for one physical parameter set."""
    assert _SENS_CTX is not None
    est_qpos, est_qvel = _sens_simulate(phys)
    gt_qpos, gt_qvel = _SENS_CTX["gt_qpos"], _SENS_CTX["gt_qvel"]
    n = min(len(gt_qpos), len(est_qpos))
    w = np.linspace(0.5, 1.5, n).reshape(-1, 1)
    mse_pos = np.mean(w * (gt_qpos[:n] - est_qpos[:n]) ** 2)
    mse_vel = np.mean(w * (gt_qvel[:n] - est_qvel[:n]) ** 2)
    if not (np.isfinite(mse_pos) and np.isfinite(mse_vel)):
        return 1e6
    return float(mse_pos + 0.5 * mse_vel)


def _sens_dq(phys) -> np.ndarray:
    """Trajectory deviation from the ground truth in degrees (flattened)."""
    assert _SENS_CTX is not None
    est_qpos, _ = _sens_simulate(phys)
    gt_qpos = _SENS_CTX["gt_qpos"]
    n = min(len(gt_qpos), len(est_qpos))
    return np.rad2deg(est_qpos[:n] - gt_qpos[:n]).ravel()


def _scaled(gt: GroundTruth, group: str, factor: float,
            index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ground-truth parameter set with one group (or element) scaled."""
    vals = {"stiffness": gt.stiffness.copy(),
            "damping": gt.damping.copy(),
            "tendon_stiffness": gt.tendon_stiffness.copy()}
    if index is None:
        vals[group] *= factor
    else:
        vals[group][index] *= factor
    return vals["stiffness"], vals["damping"], vals["tendon_stiffness"]


def run_sensitivity(sim_time: float, init_angle_deg: float, workers: int, seed: int,
                    n_oat: int = 40, n_grid: int = 41, delta: float = 0.10,
                    span: tuple[float, float] = (0.4, 2.5)) -> dict[str, Any]:
    """Full sensitivity analysis for one scenario."""
    global _SENS_CTX

    print("\n" + "=" * 60)
    print(f"SENSITIVITY ANALYSIS  sim_time={sim_time}s, start pose={init_angle_deg} deg")
    print("=" * 60)
    gt = build_ground_truth(sim_time, init_angle_deg, seed=seed, verbose=False)
    _SENS_CTX = {"sim_time": sim_time, "init_angle_deg": init_angle_deg,
                 "gt_qpos": gt.qpos, "gt_qvel": gt.qvel}

    pool = None
    if workers and workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=workers, initializer=_pool_init_ignore_sigint)

    def pmap(fn, tasks):
        if pool is not None:
            return pool.map(fn, tasks)
        return [fn(t) for t in tasks]

    try:
        # Sanity check: at the ground-truth point the cost must be exactly zero.
        cost_gt = _sens_cost(_scaled(gt, "stiffness", 1.0))
        print(f"  check: cost at the ground-truth point = {cost_gt:.3e}")

        # ── 1) OAT scan: cost over relative deviation, per group ─────────
        # an even number of samples drops the factor 1.0 (where cost = 0)
        factors = np.logspace(np.log10(span[0]), np.log10(span[1]), n_oat)
        oat: dict[str, list[float]] = {}
        for g in SENS_GROUPS:
            costs = pmap(_sens_cost, [_scaled(gt, g, f) for f in factors])
            oat[g] = [float(c) for c in costs]
            print(f"  [OAT] {g:>17s}: cost at -50 % / +100 % = "
                  f"{np.interp(0.5, factors, costs):.3e} / {np.interp(2.0, factors, costs):.3e}")

        # ── 2) Trajectory sensitivity per individual parameter ───────────
        # S = RMS angle deviation [deg] for a +delta relative change,
        # centrally averaged over +delta and -delta.
        traj_sens: dict[str, list[float]] = {}
        for g in SENS_GROUPS:
            n_el = len(getattr(gt, g if g != "tendon_stiffness" else "tendon_stiffness"))
            tasks = []
            for i in range(n_el):
                tasks.append(_scaled(gt, g, 1.0 + delta, index=i))
                tasks.append(_scaled(gt, g, 1.0 - delta, index=i))
            dqs = pmap(_sens_dq, tasks)
            s_vals = []
            for i in range(n_el):
                rms_p = float(np.sqrt(np.mean(dqs[2 * i] ** 2)))
                rms_m = float(np.sqrt(np.mean(dqs[2 * i + 1] ** 2)))
                s_vals.append(0.5 * (rms_p + rms_m))
            traj_sens[g] = s_vals
            print(f"  [Sens] {g:>17s}: {np.min(s_vals):.4f} … {np.max(s_vals):.4f} deg "
                  f"per {delta*100:.0f} % change")

        # ── 3) Group sensitivity + collinearity ─────────────────────────
        group_dq = {}
        for g in SENS_GROUPS:
            group_dq[g] = _sens_dq(_scaled(gt, g, 1.0 + delta))
        group_sens = {g: float(np.sqrt(np.mean(group_dq[g] ** 2))) for g in SENS_GROUPS}
        collin = np.zeros((len(SENS_GROUPS), len(SENS_GROUPS)))
        for a, ga in enumerate(SENS_GROUPS):
            for b, gb in enumerate(SENS_GROUPS):
                va, vb = group_dq[ga], group_dq[gb]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                collin[a, b] = float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0
        print("  [group] total sensitivity: "
              + ", ".join(f"{g}={group_sens[g]:.4f}°" for g in SENS_GROUPS))
        print(f"  [collinearity] stiffness<->damping = {collin[0,1]:+.3f}")

        # ── 3b) Cost rise over logarithmically spaced deviations ─────────
        # Covers 0.01 % ... 150 % and shows from which deviation onwards the
        # cost function still separates parameter sets at all.
        dev_log = np.logspace(-4, np.log10(1.5), 25)
        oat_log: dict[str, list[float]] = {}
        for g in SENS_GROUPS:
            costs = pmap(_sens_cost, [_scaled(gt, g, 1.0 + d) for d in dev_log])
            oat_log[g] = [float(c) for c in costs]
        print("  [OAT-log] cost at a 0.01 % deviation: "
              + ", ".join(f"{g}={oat_log[g][0]:.4f}" for g in SENS_GROUPS))

        # ── 3c) Trajectory divergence under tiny perturbations ───────────
        # If the deviation grows exponentially and saturates, the system is
        # chaotic at this operating point, and a longer trajectory carries no
        # further parameter information.
        div_perturb = [1e-6, 1e-4, 1e-2]
        divergence = {}
        for d in div_perturb:
            qp, _ = _sens_simulate(_scaled(gt, "stiffness", 1.0 + d))
            n = min(len(qp), len(gt.qpos))
            dq = np.rad2deg(qp[:n] - gt.qpos[:n])
            divergence[f"{d:g}"] = np.sqrt(np.mean(dq ** 2, axis=1)).tolist()
        t_axis = (np.arange(len(next(iter(divergence.values())))) * 0.1).tolist()
        print(f"  [divergence] final deviation for 1e-6 / 1e-2 perturbation: "
              f"{divergence['1e-06'][-1]:.3f}° / {divergence['0.01'][-1]:.3f}°")

        # ── 3d) Cost over the length of the fitting horizon ──────────────
        # How long may the evaluated time window be before the cost runs into
        # the same plateau regardless of perturbation size?
        hor_perturb = [1e-3, 1e-2, 1e-1]
        horizon_costs = {}
        n_full = len(gt.qpos)
        t_grid = np.arange(2, n_full + 1)
        for d in hor_perturb:
            qp, qv = _sens_simulate(_scaled(gt, "stiffness", 1.0 + d))
            vals = []
            for k in t_grid:
                m = min(k, len(qp))
                w = np.linspace(0.5, 1.5, m).reshape(-1, 1)
                mp = np.mean(w * (gt.qpos[:m] - qp[:m]) ** 2)
                mv = np.mean(w * (gt.qvel[:m] - qv[:m]) ** 2)
                vals.append(float(mp + 0.5 * mv))
            horizon_costs[f"{d:g}"] = vals
        horizon_t = (t_grid * 0.1).tolist()

        # ── 4) 2-D cost landscape: stiffness x damping ───────────────────
        gf = np.logspace(np.log10(span[0]), np.log10(span[1]), n_grid)
        tasks = []
        for fs in gf:
            for fd in gf:
                tasks.append((gt.stiffness * fs, gt.damping * fd, gt.tendon_stiffness.copy()))
        land = np.array(pmap(_sens_cost, tasks), dtype=float).reshape(n_grid, n_grid)
        print(f"  [landscape] {n_grid}x{n_grid} grid, cost {land.min():.3e} ... {land.max():.3e}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return {
        "sim_time": sim_time,
        "init_angle_deg": init_angle_deg,
        "delta": delta,
        "cost_at_gt": cost_gt,
        "factors": factors.tolist(),
        "oat": oat,
        "dev_log": dev_log.tolist(),
        "oat_log": oat_log,
        "divergence_time": t_axis,
        "divergence": divergence,
        "horizon_time": horizon_t,
        "horizon_costs": horizon_costs,
        "traj_sens": traj_sens,
        "group_sens": group_sens,
        "collinearity": collin.tolist(),
        "collinearity_labels": list(SENS_GROUPS),
        "landscape_factors": gf.tolist(),
        "landscape": land.tolist(),
        "ground_truth": {
            "stiffness": gt.stiffness.tolist(),
            "damping": gt.damping.tolist(),
            "tendon_stiffness": gt.tendon_stiffness.tolist(),
        },
    }


# ====================================================================
#  Plots for single runs
# ====================================================================

def plot_single_run(record: dict[str, Any], gt: GroundTruth, sim_time: float,
                    init_angle_deg: float, suffix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [Warn] plots not possible: {exc}")
        return

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    hist = record["cost_history"]
    if hist:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(hist, marker="o", linestyle="-", linewidth=2, markersize=4, color="steelblue")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Cost (log scale)")
        ax.set_title(f"Optimization Convergence ({record['optimizer'].upper()})")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        p = BUILD_DIR / f"sysid_multi_convergence{suffix}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [Info] convergence plot saved to: {p}")

    try:
        opt_model = make_model()
        ident = record["identified"]
        set_params(opt_model, np.array(ident["stiffness"]), np.array(ident["damping"]),
                   np.array(ident["tendon_stiffness"]))
        est_qpos, _ = simulate(opt_model, sim_time, init_angle_deg=init_angle_deg)

        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, gt.njnt))
        for i in range(gt.njnt):
            axs[0].plot(gt.qpos[:, i], label=f"J{i}", color=colors[i])
        axs[0].set_title("Ground Truth Trajectories")
        axs[0].set_ylabel("qpos (rad)")
        axs[0].grid(True, linestyle="--", alpha=0.5)
        for i in range(gt.njnt):
            axs[1].plot(est_qpos[:, i], label=f"J{i}", color=colors[i])
        axs[1].set_title("Simulated Trajectories (Identified Parameters)")
        axs[1].set_xlabel("Samples")
        axs[1].set_ylabel("qpos (rad)")
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        p = BUILD_DIR / f"sysid_multi_validation{suffix}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [Info] validation plot saved to: {p}")
    except Exception as exc:
        print(f"\n  [Warn] validation plot could not be created: {exc}")


# ====================================================================
#  Main
# ====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-Parameter SpiRob System-ID (Sim-to-Sim)")
    ap.add_argument("--sim-time", type=float, default=2.0, help="Length of the recorded trajectory [s]")
    ap.add_argument("--maxiter", type=int, default=50, help="Number of optimizer generations")
    ap.add_argument("--tol", type=float, default=0.2, help="Convergence tolerance (DE only)")
    ap.add_argument("--workers", type=int, default=10,
                    help="Number of parallel worker processes")
    ap.add_argument("--init-angle", type=float, default=INIT_JOINT_ANGLE_DEG,
                    help="Start pose: angle on every joint [deg] (0 = rest pose)")
    ap.add_argument("--optimizer", choices=["de", "cma"], default="de",
                    help="Optimierungsverfahren (Default: de)")
    ap.add_argument("--cma-sigma0", type=float, default=0.25,
                    help="Initial CMA-ES step size in the normalized box")
    ap.add_argument("--seed", type=int, default=42, help="Seed of the ground-truth scatter")
    ap.add_argument("--compare", action="store_true",
                    help="Compare DE and CMA-ES across all configurations")
    ap.add_argument("--configs", type=str, default=None,
                    help="Configurations for --compare: 'simtime:maxiter:angle,...' "
                         "(default: the five configurations from the thesis)")
    ap.add_argument("--budget-mode", choices=["evals", "generations"], default="evals",
                    help="Budgetangleich im Vergleichsmodus: gleiche Funktionsauswertungen "
                         "(default, fair) or the same generation count")
    ap.add_argument("--sensitivity", action="store_true",
                    help="Run the sensitivity/identifiability analysis instead of an optimization")
    ap.add_argument("--sens-scenarios", type=str, default="4:0,20:0,10:20",
                    help="Scenarios of the sensitivity analysis: 'simtime:angle,...'")
    ap.add_argument("--sens-delta", type=float, default=0.10,
                    help="Relative parameter change used for the trajectory sensitivity")
    ap.add_argument("--sens-grid", type=int, default=41,
                    help="Resolution of the 2-D cost landscape, per axis")
    ap.add_argument("--out", type=str, default=None,
                    help="Path of the result JSON (default: build/sim2sim/sysid_multi_<mode>.json)")
    ap.add_argument("--profile-timing", action="store_true")
    args = ap.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if args.sensitivity:
        scenarios = []
        for tok in args.sens_scenarios.split(","):
            tok = tok.strip()
            if not tok:
                continue
            parts = tok.split(":")
            if len(parts) != 2:
                raise SystemExit(f"--sens-scenarios: expected simtime:angle, got '{tok}'")
            scenarios.append((float(parts[0]), float(parts[1])))

        results = [run_sensitivity(st, ang, args.workers, args.seed,
                                   delta=args.sens_delta, n_grid=args.sens_grid)
                   for st, ang in scenarios]

        out = Path(args.out) if args.out else BUILD_DIR / "sysid_sensitivity.json"
        out.write_text(json.dumps({"seed": args.seed, "scenarios": results}, indent=2))
        print(f"\n  [Info] sensitivity results saved to: {out}")
        return

    if args.compare:
        configs = parse_configs(args.configs)
        records = run_comparison(configs, args.workers, args.tol, args.cma_sigma0,
                                 args.budget_mode, args.seed)
        print_comparison_summary(records)

        out = Path(args.out) if args.out else BUILD_DIR / "sysid_multi_de_vs_cma.json"
        out.write_text(json.dumps({"budget_mode": args.budget_mode, "workers": args.workers,
                                   "seed": args.seed, "records": records}, indent=2))
        print(f"\n  [Info] results saved to: {out}")

        plot_comparison(records, BUILD_DIR / "sysid_multi_de_vs_cma.png")

        typ = typst_table(records)
        typ_path = BUILD_DIR / "sysid_multi_de_vs_cma.typ"
        typ_path.write_text(typ)
        print(f"  [Info] Typst table saved to: {typ_path}\n")
        print(typ)
        return

    print("=" * 55)
    print("Multi-Parameter Ground-Truth simulieren …")
    gt = build_ground_truth(args.sim_time, args.init_angle, seed=args.seed, verbose=True)

    print("\n" + "=" * 55 + f"\nOptimierung starten ({args.optimizer})\n" + "-" * 55)
    record = run_experiment(args.sim_time, args.maxiter, args.init_angle, args.optimizer,
                            args.workers, args.tol, args.cma_sigma0, gt=gt, verbose=True,
                            seed=args.seed)

    out = Path(args.out) if args.out else BUILD_DIR / f"sysid_multi_{args.optimizer}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\n  [Info] result saved to: {out}")

    plot_single_run(record, gt, args.sim_time, args.init_angle, f"_{args.optimizer}")


if __name__ == "__main__":
    main()
