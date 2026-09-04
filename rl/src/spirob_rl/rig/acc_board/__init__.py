"""Joint-angle estimation from the SpiRob accelerometer board.

``frame_parser`` and ``kinematic_optimization`` are self-contained (stdlib +
numpy/scipy, no cross-imports), so they import cleanly as a subpackage. The
live/visualisation scripts here use same-directory absolute imports and are
meant to be run directly, not imported.
"""
