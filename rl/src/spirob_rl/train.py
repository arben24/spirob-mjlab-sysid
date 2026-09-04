"""Training entrypoint. Registers the SpiRob tasks, then runs mjlab's trainer."""

from __future__ import annotations

from .cli import main_train


def main() -> None:
  main_train()


if __name__ == "__main__":
  main()
