# Tools

Interactive inspection of the identified model.

## `live_dashboard.py`

Opens the MuJoCo viewer with the identified parameters already applied, plus
live plots of everything MuJoCo exposes while you drive the robot with the
actuator sliders:

* joint angles `q` (`mjData.qpos`) and velocities `v` (`qvel`) over time
* tip position in the x–z plane, as a 2-D trajectory
* mass matrix `M(q)` (`qM`, densified) as a heatmap
* bias force `c(q,v)` (`qfrc_bias` — Coriolis + centrifugal + gravity)
* applied force `τ` (`qfrc_passive + qfrc_actuator + qfrc_applied`)
* constraint forces `f` (`efc_force`)

The realtime factor is shown in the title.

```bash
uv pip install -e ".[gui]"

uv run sysid/tools/live_dashboard.py
uv run sysid/tools/live_dashboard.py --xml models/spirob_13seg.xml --no-params
```

**Why it is threaded:** rendering a full Matplotlib frame costs 60–80 ms. If the
physics were stepped in the same loop, every redraw would stall the simulation.
Instead the physics and the viewer run in a background thread with catch-up
pacing — a step is ~475× faster than realtime, so the sim sprints through any
backlog a draw caused.

Needs a display. It will not run headless.
