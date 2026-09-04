"""Auto-discover and import every SpiRob task package below this one."""

from __future__ import annotations

import importlib
import pkgutil


def import_all_tasks() -> None:
  package = importlib.import_module(__name__)
  prefix = package.__name__ + "."
  for module_info in pkgutil.walk_packages(package.__path__, prefix):
    if module_info.name.rsplit(".", 1)[-1].startswith("_"):
      continue
    importlib.import_module(module_info.name)


import_all_tasks()
