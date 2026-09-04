"""SpiRob reinforcement learning: mjlab tasks, entrypoints and the rig bridge.

Importing this package registers every task with mjlab's registry (see
``tasks/__init__.py``), which is what makes ``train``/``play``/``infer`` able to
resolve a task id.
"""

from . import tasks  # noqa: F401
