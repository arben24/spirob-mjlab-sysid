"""Canonical locations inside the repository.

Every script writes its output to :data:`BUILD_DIR` and reads its input from
:data:`DATA_DIR` / :data:`MODELS_DIR`, so nothing ever lands next to the source
files and ``build/`` can be wiped without losing anything tracked.

The root is derived from this file's location (the package is installed
editable from ``src/``). Set ``SPIROB_ROOT`` to override, e.g. when the package
is installed as a wheel outside the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("SPIROB_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = REPO_ROOT / "data"
"""Measured data. Read-only — nothing in the repo writes here."""

MODELS_DIR = REPO_ROOT / "models"
"""Tracked MuJoCo XML models."""

BUILD_DIR = Path(os.environ.get("SPIROB_BUILD", REPO_ROOT / "build"))
"""All generated output (figures, fitted parameters, videos). Git-ignored."""

FREE_VIBRATION_DIR = DATA_DIR / "free_vibration"
STATIC_LOAD_DIR = DATA_DIR / "static_load"
TRAJECTORY_DIR = DATA_DIR / "trajectories"
IDENTIFIED_DIR = DATA_DIR / "identified"

DEFAULT_MODEL = MODELS_DIR / "spirob_13seg.xml"
"""Nominal 13-joint model — the starting point of every identification run."""

IDENTIFIED_MODEL = MODELS_DIR / "spirob_13seg_identified.xml"
"""Same model with the real-to-sim CMA-ES parameters baked in."""

DEFAULT_TRAJECTORY = TRAJECTORY_DIR / "spirob_tendon_trajectory_60s.parquet"
"""60 s reference recording: tendon forces + ArUco joint angles."""

DEFAULT_PARAMS = IDENTIFIED_DIR / "real2sim_cma_500iter.json"
"""Final identified parameter set (CMA-ES, 500 iterations)."""


def build_dir(*parts: str) -> Path:
    """Return ``BUILD_DIR/parts`` and create it (directories only)."""
    p = BUILD_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "BUILD_DIR",
    "FREE_VIBRATION_DIR",
    "STATIC_LOAD_DIR",
    "TRAJECTORY_DIR",
    "IDENTIFIED_DIR",
    "DEFAULT_MODEL",
    "IDENTIFIED_MODEL",
    "DEFAULT_TRAJECTORY",
    "DEFAULT_PARAMS",
    "build_dir",
]
