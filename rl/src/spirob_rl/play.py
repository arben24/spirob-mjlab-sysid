"""Playback entrypoint: roll out a trained checkpoint in the simulator."""

from __future__ import annotations

from .cli import main_play


def main() -> None:
  main_play()


if __name__ == "__main__":
  main()
