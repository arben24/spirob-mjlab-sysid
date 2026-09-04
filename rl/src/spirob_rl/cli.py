"""Shared CLI helpers for the SpiRob RL entrypoints.

Thin wrappers around mjlab's own ``train``/``play`` scripts. Importing
:mod:`spirob_rl.tasks` is the only thing that has to happen first: that is what
registers the task ids the scripts then look up.

The one behavioural addition is the log root. mjlab writes checkpoints to
``logs/rsl_rl`` relative to the current working directory, which would scatter
runs across wherever the command was typed. This repository has exactly one
place for generated output (``build/``, git-ignored, see ``spirob.paths``), so
an unspecified ``--log-root`` is filled in with ``build/rl/logs``.
"""

from __future__ import annotations

import sys

from mjlab.scripts.play import main as mjlab_play_main
from mjlab.scripts.train import main as mjlab_train_main
from spirob.paths import build_dir

from . import tasks as _tasks  # noqa: F401

LOG_ROOT = build_dir("rl", "logs")
"""Where training runs land: ``build/rl/logs/<experiment_name>/<timestamp>/``."""


def _apply_default_log_root() -> None:
  """Insert ``--log-root build/rl/logs`` unless the caller set one.

  Done on ``sys.argv`` rather than by patching the config dataclass: mjlab's
  ``TrainConfig``/``PlayConfig`` are frozen and parsed by tyro inside its own
  ``main()``, so the command line is the only seam that does not depend on
  mjlab internals staying put.
  """
  if any(arg == "--log-root" or arg.startswith("--log-root=") for arg in sys.argv[1:]):
    return
  sys.argv.extend(["--log-root", str(LOG_ROOT)])


def main_train() -> None:
  _apply_default_log_root()
  mjlab_train_main()


def main_play() -> None:
  _apply_default_log_root()
  mjlab_play_main()


def main_run() -> None:
  main_play()
