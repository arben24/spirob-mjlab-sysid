"""The SpiRob rig: everything outside the training loop.

Two groups of modules, both consuming a policy trained by
:mod:`spirob_rl.tasks` rather than being part of it:

* **Hardware bridge** — ``serial_protocol``/``sources`` talk to the two ESP32
  boards, ``observation`` rebuilds the exact observation vector the task's
  actor was trained on, ``policy_bridge`` closes the loop, ``telemetry`` is the
  UDP side channel and ``target_gui`` the operator front end.
* **Analysis** — ``workspace_sweep`` drives a trained reach policy across a
  grid of targets in simulation and ``workspace_figure`` draws the reachability
  map; ``target_figure``/``target_crop`` turn a recorded reach into a
  publication figure.

Nothing here is imported by the tasks; the dependency runs one way only.
"""
