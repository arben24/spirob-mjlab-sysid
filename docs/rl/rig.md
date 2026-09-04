# From simulation to the rig

## TL;DR

A trained policy leaves simulation in three hops, each deliberately its own
process:

1. **`policy_bridge`** loads a checkpoint, rebuilds the observation vector from
   live sensor frames and writes tendon forces back over serial at 50 Hz.
2. **`target_gui`** sets the target and shows the measured tip — a separate
   process talking UDP, so a slow GUI can never stall the control loop.
3. **`target_figure` / `workspace_figure`** redraw the stored numbers as
   publication figures, never as screenshots.

```bash
cd rl
uv sync --extra hardware
uv run python -m spirob_rl.rig.policy_bridge RlExplor-Spirob-Tcp-Reach-Imu \
    --port /dev/ttyUSB0 --joint-port /dev/ttyUSB1 --dry-run
uv run python -m spirob_rl.rig.target_gui
```

Start every first run with `--dry-run`, bend the arm by hand and check the
reported joint angles before a motor ever turns.

## Which policies can actually run on hardware

The sensor level a policy was trained at decides what the bridge must be able to
supply:

| Task id | Level | Needs | Runs on the rig |
|---|---|---|---|
| `…-Tcp-Reach-Force` | force | nothing but target and last action | ✓ motor board alone |
| `…-Tcp-Reach` | tendon | tendon length + velocity | ✓ motor board alone |
| `…-Tcp-Reach-Imu` | imu | + inclination of all 14 segments | ✓ motor + accelerometer board |
| `…-Tcp-Reach-Joints` | joints | 13 joint angles + velocities | ✓ motor + accelerometer board |
| `…-Tcp-Reach-Oracle` | oracle | privileged TCP position | ✗ refused at startup |

Only `oracle` is impossible, and the bridge says so before anything moves.

The reason `imu` is realisable is worth spelling out: the observation term is
cos/sin of each segment's *absolute* inclination, and that is precisely what an
accelerometer measures in a gravity field. The board's solver reads the
inclinations as `atan2(a_z, a_y)` and reports their differences as joint angles;
the gravity reference and any tilt of the whole rig cancel out of those
differences, so beyond sign and ordering there is nothing to calibrate.

## Keeping the observation honest

The bridge does not hard-code an observation layout. It builds the task
environment once at startup and reads the actor group's term order, per-term
width and history depth straight off mjlab's observation manager, then
reassembles that exact vector from live measurements — history stacked
oldest-first per term, the first frame backfilled, matching how mjlab's
`CircularBuffer` fills.

That matters because nothing about a wrong layout raises an error. A
transposed history or a swapped pair of terms still produces plausible numbers,
and the rig simply moves wrong. Reading the layout from the task means the
task's own `history_length` moves both sides at once.

For an `-Imu` policy the vector is 185 numbers — five frames of:

| Term | Dim | Source on the rig |
|---|---|---|
| `target` | 3 | GUI, stdin or `--target-x/--target-z` |
| `last_action` | 2 | the bridge's own previous output |
| `tendon_len` | 2 | spool encoders on the motor board |
| `tendon_vel` | 2 | the same, filtered and differentiated |
| `segment_pitch` | 28 | 14 segment inclinations as cos/sin |

## Commanded vs. measured

The target is what you set; where the arm actually is comes from the
accelerometer board. The rig cannot measure its tip directly, but it is a chain
of rigid links of known length, so the 13 joint angles give the TCP position —
with the segment lengths read at runtime from the same MJCF the policy trained
on, never copied into a constant.

That makes the quantity the reward is built on — `|TCP − target|` — measurable
on hardware, even though the task treats TCP position as privileged
information the actor is not allowed to see.

## Telemetry and figures

The bridge publishes one JSON-over-UDP datagram per control tick (newest
observation frame, action, commanded target, hardware-frame angles, loop
timing). The GUI subscribes to it. The split is not cosmetic: the control loop
has a 20 ms budget, so the observer runs as its own process and a dead or slow
GUI cannot apply backpressure.

Figures are a third hop and deliberately not screenshots. `target_figure` takes
the raw numbers the GUI already holds — target, measured tip, arm pose, recorded
tip path — as a serialisable scene and redraws them with matplotlib. Every
capture writes a PDF, a PNG and the scene JSON into `build/rl/figures/`, so a
figure can be re-rendered later in a different language or size without going
back to the rig. `target_crop` is the drag-to-select GUI for trimming the tip
path afterwards.

## The workspace map

The reach task draws one random target per episode, so training and play only
ever sample the workspace: a good run says the policy works *somewhere*, not
where in the shell it works well.

```bash
uv run python -m spirob_rl.rig.workspace_sweep --figure
uv run python -m spirob_rl.rig.workspace_figure \
    --sweep ../build/rl/workspace/sweep_*.npz --threshold 30
```

`workspace_sweep` lays a grid over the task's *own* target distribution — read
off `TcpPositionCommandCfg` rather than copied — drives the policy to every
point in simulation and stores the whole distance-over-time trace per rollout.

Two design choices carry the result:

* **Every target is approached from the same seeded set of start postures.** The
  reset draws 13 independent joint offsets, and a tentacle that happens to start
  leaning toward its target has an easier time. Sampling fresh starts per target
  would fold that luck into the map and show it as a property of the target; a
  paired design turns the spread across repeats into its own readable quantity.
* **The command term's resampling is replaced, not overwritten after reset**, so
  the observation history the policy sees never describes a target that was
  never commanded.

Because the raw distances are stored, the success threshold is a rendering
decision (`--threshold`), not a reason to re-run the sweep.

## Firmware

The ESP32 side — the motor controller, the accelerometer board and the onboard
policy variant that runs inference on the MCU itself — lives in the RL_explor
repository this task family came from and is not duplicated here. The host half
of the protocol is documented in
`rl/src/spirob_rl/rig/COMMUNICATION_PROTOCOL.md`; if the protocol changes, both
sides have to change together.
