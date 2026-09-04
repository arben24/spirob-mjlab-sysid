# The task family

## TL;DR

One tentacle, four goals, five levels of actor observability — 20 tasks, each
with a `-DrPlay` twin that replays a trained policy under full-width domain
randomisation, so 40 registered ids.

| Variant | Task id prefix | Goal |
|---|---|---|
| Reach | `RlExplor-Spirob-Tcp-Reach` | hold the TCP at a static random target on the reachable shell |
| Shape | `RlExplor-Spirob-Shape` | hit a TCP *and* a mid-chain target — command the whole posture |
| Trajectory | `RlExplor-Spirob-Trajectory` | follow a target sweeping along the arc, with preview points |
| Wrap | `RlExplor-Spirob-Wrap` | coil around a randomly placed, randomly sized cylinder |

Everything that is *not* the goal — the entity, the action, the observation
ladder, domain randomisation, the grid layout, the solver settings, the PPO
configuration — lives once in `base_env_cfg.py`. Each variant file contributes
only its command and reward terms.

## What every variant shares

**Action.** Two tendon force actuators. The policy's `[-1, 1]` maps onto
`[-150, 0] N` (`scale=75`, `offset=-75`) — **pull only**, a positive control
value is a silent no-op. A hard `clip_actions=5.0` sits in front of that in the
runner config, on the raw pre-scale action: without it the action-magnitude
penalty is computed on an unbounded number, and a bad gradient step can walk the
Gaussian mean out past the actuator's saturation point, where nothing in the
physics pulls it back. That was observed to explode the value loss to `inf`
within ten iterations on the wrap task.

**Simulation.** `timestep = 0.004 s`, elliptic cone, `impratio = 18.78`, 20
solver iterations — read off the `<option>` element of the model — with
`decimation = 5`, i.e. a 50 Hz policy.

**Episode.** 50 s for reach/shape/trajectory, 30 s for wrap. The only
termination is the time-out; there is no failure state to fall into.

**Reset.** Each of the 13 joints independently draws its own offset from
`(-0.5, 0.5) rad`. The wrap task narrows this to `(0.0, 0.5)` so the reset pose
can never reach into the half-space the object spawns in.

## The sensor ladder

Which observations the *actor* gets is the experiment. The critic always
receives joint angles, joint velocities, TCP position, the target and the last
action, so a comparison across levels isolates the effect of the actor's sensor
suite rather than of the value function.

| Suffix | Level | Actor observation | Realisable on the rig |
|---|---|---|---|
| `-Force` | `force` | target + last action | ✓ motor board alone |
| *(none)* | `tendon` | + tendon length and velocity (spool encoders) | ✓ motor board alone |
| `-Imu` | `imu` | + cos/sin of all 14 segment inclinations | ✓ motor + accelerometer board |
| `-Joints` | `joints` | 13 joint angles and velocities | ✓ motor + accelerometer board |
| `-Oracle` | `oracle` | + TCP position | ✗ not measurable |

Two subtleties are worth stating explicitly:

* At the `force` level the observation is *the commanded action itself* (tendon
  force equals the control, verified against `tendonactuatorfrc`), so that rung
  is effectively "no state feedback" and depends entirely on the observation
  history.
* `joints` is realisable even though it looks privileged: each adjacent pair of
  accelerometers on the rig's angle board yields one joint angle, so the sim's
  `joint_pos` is measurable. `oracle` differs from it only by `tcp_pos`, which
  the rig genuinely cannot sense.

The actor group is stacked with a **history** (5 frames in play, 10 in the
training config's default) and has observation corruption enabled. History is
not a nicety here: a tendon-length sample does not determine the shape of a
13-joint chain, so a single frame is not a state.

## Commands

All commands are point-tracking commands in the x–z bending plane: they own
`target_pos_w` and the matching measured `tracked_pos_w`, both
`[num_envs, num_points, 3]`, and one generic Gaussian-kernel reward works off
either.

**Reach** samples a target on the *reachable shell* — the tentacle is
inextensible, so its quasi-static workspace is a thin arc, parameterised as
`r(angle) = 0.33 − 0.045 · angle²` with a ±0.08 m band over `angle ∈ [−1.7, 1.7]`
rad. Those numbers are fitted to a quasi-static random-actuation probe, not
guessed.

**Shape** needs two targets that are achievable *together*. Two independently
drawn points almost never lie on one pose of an inextensible, two-tendon chain,
so instead a joint configuration is drawn and both targets are read off its
forward kinematics. The configurations come from `holdable_poses.npz`: with 13
joints but only two tendon forces, the statically holdable poses form at most a
2-manifold in joint space, and analytic guesses at that manifold fit badly
(~0.2 rad residual per joint against a 0.51 rad joint limit). So it is measured
— a grid over the two tendon commands, each let settle, the resulting pose
recorded — and sampling bilinearly interpolates inside that grid.

**Trajectory** sweeps the target angle as a sinusoid (amplitude 0.4–1.5 rad,
0.05–0.25 Hz) with the radius riding the same shell, so the commanded point is
reachable at every instant. It also reports three preview points 0.2 s apart,
which is what lets a policy lead the target instead of chasing it.

**Wrap** is the odd one out: the "target" is a real collidable cylinder placed
per environment via a mocap body, with its radius written into the compiled
model's `geom_size` (plus `geom_rbound`/`geom_aabb`, which MuJoCo-Warp does not
recompute on its own — leaving them stale silently drops contacts for the larger
objects). It resamples **only** at reset, never mid-episode: a new abstract
point mid-episode is harmless, a new cylinder materialising around the tentacle
is not.

## Rewards

Reach and trajectory use a coarse and a fine Gaussian kernel on the
target-to-TCP distance (`std = 0.2` at weight 1, `std = 0.05` at weight 3) plus
small penalties on action rate, action magnitude and joint velocity. Shape adds
a separate mid-segment term, deliberately looser than the tip's — the tip is
what the task is about, and an over-tight mid term would fight it for the same
two tendons.

Wrap needs more than distance, because "wrapped" is not "touching":

| Term | What it measures |
|---|---|
| `wrap_proximity` (coarse + fine) | Gaussian kernel on every segment site's distance to the cylinder surface, averaged — the whole body hugging it, not just the tip |
| `wrap_coverage` | circular-statistics resultant length of the close segments around the object: 0 when they cluster on one side, growing as the wrap actually encircles it |
| `wrap_force_distribution` | normalised Shannon entropy of the per-segment **contact force** from a `ContactSensor` — rewards spreading the load, not just the geometry |

The last one exists because the three geometric terms can all be satisfied by
twelve segments grazing the surface while one does the gripping.

## Domain randomisation

Randomisation is enabled by default and ramped in by a curriculum, both
controlled from the top of `base_env_cfg.py`:

```python
ENABLE_DOMAIN_RANDOMIZATION = True   # False -> nominal XML dynamics everywhere
ENABLE_DR_CURRICULUM = True          # False -> full width from step 0
```

Active terms and the widths they ramp toward (`DR_TARGETS` in
`mdp/constants.py`), all with `operation="scale"` against the XML default:

| Term | Target width |
|---|---|
| `joint_stiffness` | ×[0.5, 1.0] |
| `joint_damping` | ×[0.1, 1.0] |
| `joint_friction` | ×[0.1, 1.0] |
| `tendon_stiffness` | ×[0.5, 1.0] |
| `tendon_damping` | ×[0.5, 1.0] |
| `tendon_frictionloss` | ×[0.5, 1.0] |

`scale` rather than an absolute range is a deliberate choice: the identified
model's per-joint damping spans four orders of magnitude, and an absolute band
would collapse that structure. The XML value is therefore each joint's upper
bound and the draw reaches down to `lo × default`. The curriculum interpolates
each term from the no-op width `(1.0, 1.0)` to its target over the first 5000
policy steps, keyed on `env.common_step_counter`, so training starts on the
nominal model and the spread grows as the policy stabilises.

Which parameters get the widest bands follows the identifiability result from
the identification: joint stiffness was recovered to only 28–158 % even
sim-to-sim, tendon stiffness to better than 5 %. Randomise each parameter in
proportion to how poorly it is known.
