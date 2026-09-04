"""Alias for :mod:`spirob_rl.play`, kept so ``uv run run`` behaves as expected."""

from __future__ import annotations

from mjlab.scripts._cli import maybe_print_top_level_help

from .cli import main_run


def main() -> None:
  maybe_print_top_level_help("run")
  main_run()


if __name__ == "__main__":
  main()
