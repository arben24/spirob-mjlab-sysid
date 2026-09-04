#!/usr/bin/env python3
"""Render a headless MuJoCo video of the SpiRob curling.

Two sources of excitation:

``--controller sine``  (default) a smooth alternating pull on the two tendons —
    a short, self-contained clip that needs nothing but the model.
``--controller replay`` the *measured* tendon forces from
    ``data/trajectories/*.parquet`` — the same excitation the identification was
    fitted against, so the clip shows what the identified model actually does.

Rendering is offscreen through EGL, so this runs on a headless machine. If EGL
is unavailable, set ``MUJOCO_GL=osmesa`` for a software fallback.

Needs the ``vision`` extra for the video encoder::

    uv pip install -e ".[vision]"

Usage::

    uv run scripts/render_demo.py
    uv run scripts/render_demo.py --model models/spirob_13seg.xml --seconds 6
    uv run scripts/render_demo.py --controller replay --seconds 30 --gif

Outputs: build/media/spirob_demo.mp4  (+ .gif with --gif)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco as mj  # noqa: E402
import numpy as np  # noqa: E402

from spirob.paths import DEFAULT_TRAJECTORY, MODELS_DIR, build_dir  # noqa: E402

DEMO_SCENE = MODELS_DIR / "scene_demo.xml"


def replay_forces(seconds: float, dt: float) -> np.ndarray:
    """Measured tendon forces resampled onto the simulation time grid, shape (n, 2)."""
    import polars as pl

    df = pl.read_parquet(DEFAULT_TRAJECTORY)
    t = df["global_timestamp_s"].to_numpy()
    t = t - t[0]
    f = np.column_stack([df["meas_force_0_N"].to_numpy(), df["meas_force_1_N"].to_numpy()])
    grid = np.arange(0.0, seconds, dt)
    return np.column_stack([np.interp(grid, t, f[:, i]) for i in range(2)])


def _apply_ctrl(data, forces, i, t, args) -> None:
    """Set the two tendon controls. ctrlrange is pull-only, hence the sign."""
    if forces is not None:
        data.ctrl[0] = -abs(float(forces[i, 0]))
        data.ctrl[1] = -abs(float(forces[i, 1]))
    else:
        phase = 2 * np.pi * t / args.period
        data.ctrl[0] = -args.amplitude * max(0.0, float(np.sin(phase)))
        data.ctrl[1] = -args.amplitude * max(0.0, float(-np.sin(phase)))


def _sweep_extent(model, data, forces, args):
    """Bounding box of every body position over the whole clip, as (lo, hi)."""
    state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS))
    mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

    dt = model.opt.timestep
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for i in range(int(args.seconds / dt)):
        _apply_ctrl(data, forces, i, i * dt, args)
        mj.mj_step(model, data)
        if i % 25 == 0:
            xs = data.xpos[1:]  # skip the world body
            lo = np.minimum(lo, xs.min(axis=0))
            hi = np.maximum(hi, xs.max(axis=0))

    mj.mj_setState(model, data, state, mj.mjtState.mjSTATE_FULLPHYSICS)
    mj.mj_forward(model, data)
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=DEMO_SCENE,
                    help="Model XML. Default: models/scene_demo.xml (identified model + lights)")
    ap.add_argument("--controller", choices=["sine", "replay"], default="sine")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--amplitude", type=float, default=40.0, help="Peak tendon force in N (sine)")
    ap.add_argument("--period", type=float, default=4.0, help="Period of the sine pull in s")
    ap.add_argument("--azimuth", type=float, default=90.0, help="Camera azimuth in degrees")
    ap.add_argument("--elevation", type=float, default=-10.0, help="Camera elevation in degrees")
    ap.add_argument("--zoom", type=float, default=1.35,
                    help="Camera distance as a multiple of the swept diagonal")
    ap.add_argument("--gif", action="store_true", help="Also write an animated GIF")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(str(args.model))
    # The XML's default offscreen framebuffer is smaller than the video we want,
    # so widen it before the renderer allocates it.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    # The model's ground plane is a saturated blue debugging aid that swamps the
    # frame. Mute it for the video; the physics is untouched.
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[gid] = (0.33, 0.35, 0.38, 1.0)
            model.geom_size[gid][:2] = (0.6, 0.6)

    data = mj.MjData(model)
    dt = model.opt.timestep

    forces = replay_forces(args.seconds, dt) if args.controller == "replay" else None

    # Settle into the hanging equilibrium before recording, so the clip does not
    # open with the model dropping out of its XML pose.
    for _ in range(1000):
        mj.mj_step(model, data)

    # Frame the camera on the swept volume rather than on a guessed point: run
    # the excitation once, collect every body position, and centre on that box.
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = args.azimuth, args.elevation
    lo, hi = _sweep_extent(model, data, forces, args)
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = float(np.linalg.norm(hi - lo)) * args.zoom

    steps = int(args.seconds / dt)
    every = max(1, int(round(1.0 / (args.fps * dt))))
    frames = []

    renderer = mj.Renderer(model, args.height, args.width)
    try:
        for i in range(steps):
            _apply_ctrl(data, forces, i, i * dt, args)
            mj.mj_step(model, data)
            if i % every == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    finally:
        renderer.close()

    out = args.out or build_dir("media") / "spirob_demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, macro_block_size=1, quality=8)
    print(f"{len(frames)} frames -> {out}")

    if args.gif:
        gif = out.with_suffix(".gif")
        imageio.mimwrite(gif, frames[::2], fps=args.fps // 2, loop=0)
        print(f"{len(frames[::2])} frames -> {gif}")


if __name__ == "__main__":
    main()
