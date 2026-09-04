# Simulation-based identification

Instead of measuring one part, fit the *whole robot's behaviour*: excite the
real system, replay the same excitation in MuJoCo, and optimise the model
parameters until the simulated motion matches the measured motion.

## TL;DR

| Script | Scenario | Ground truth |
|---|---|---|
| `sim2sim.py` | simulation → simulation | known parameters, so the identification error is exact. Best case; an upper bound on quality. |
| `real2sim.py` | reality → simulation | ArUco-tracked joint angles of the real robot. The real problem. |
| `apply_params_to_xml.py` | — | bake a result JSON into a standalone model XML |

**Result: sim-to-sim works, real-to-sim does not.** In simulation the optimiser
recovers tendon stiffness to <5 % and damping to 4–31 %; against real data it
reproduces only the gross motion (6.8° mean RMSE) with parameters that scatter
implausibly from joint to joint. See [the method write-up](../../docs/sysid/simulation-based.md)
for why.

## Cost function

The observable is the joint-angle trajectory — it is what characterises the
whole system's behaviour and what the ArUco tracking gives us.

```
J = sqrt( Σ_ij w_i · ẽ²_pos[i,j] )  +  λ_vel · sqrt( Σ_ij w_i · ẽ²_vel[i,j] )
```

* Angle errors are converted **to degrees before squaring**. This changes
  nothing about the optimum (it scales every term equally) but keeps the cost in
  a readable range instead of a tiny one.
* `w_i` weights individual joints.
* **`λ_vel = 0.05`** is a deliberate compromise. Too large and the numerically
  bigger velocity errors dominate and position loses influence; too small and
  damping information disappears, because damping is observable mainly through
  joint velocity and barely shows in position alone. Both `w_i` and `λ_vel` were
  set empirically from sim-to-sim pre-trials, not derived.
* An unstable simulation (NaN/Inf, `|qpos| > 1e3`, MuJoCo's "huge value in
  QACC") returns a finite penalty of `1e6` rather than crashing, so DE/CMA are
  steered away without NaN-corrupting the optimizer state.

## Reducing the parameter count

Every joint could get its own stiffness, damping and friction — 13 × 3 free
variables plus tendon and solver parameters. That inflates the search space for
no good reason: the segments taper *continuously* from base to tip, so a smooth
parameter profile is the physically plausible shape.

`real2sim.py` therefore supports three representations per group:

| Representation | Meaning |
|---|---|
| `per_joint` | one independent value per element |
| `shared` | a single value applied to all elements (the default for the two tendon groups, which are symmetric) |
| `poly:N` | a degree-N polynomial along the joint chain — `N+1` coefficients instead of 13 values |

```bash
--represent "stiffness=poly:3,damping=poly:3,frictionloss=poly:2"
```

The polynomial fit did **not** reach a lower cost than independent per-joint
values, but it did not cost much either — and it produced a far more plausible
parameter distribution along the robot. That trade is usually worth taking.

## Parameter groups

`--optimize` selects which groups are free. They span four domains:

| Domain | Groups | Written to |
|---|---|---|
| joint | `stiffness`, `damping`, `frictionloss` | `jnt_stiffness`, `dof_damping`, `dof_frictionloss` (13 values) |
| tendon | `tendon_stiffness`, `tendon_damping`, `tendon_frictionloss` | `tendon_*` (2 values) |
| opt | e.g. `impratio` | `model.opt.<name>` (one scalar) |
| broadcast | `armature`, `solreflimit_*`, `solimplimit_*`, `solreffriction_*`, `solimpfriction_*` | one scalar across a whole model-array column |

Broadcast knobs are off by default; enable with `--optimize <name>`.

**Tendon seeds come from the XML** (stiffness 50, damping 0, frictionloss 0.1).
An earlier hardcoded override of 500 was far too stiff — seeding from the XML
alone dropped the validate cost by roughly 7×.

**`frictionloss` has no measurement at all**, so it gets a uniform seed of 0.15
and a wider default band.

## Search range

Per group, either a relative `band` (±%) around the seed **or** absolute
`bounds=(min,max)`. Bounds override the band.

Use absolute bounds whenever the seed is tiny. The identified damping values are
~2e-4…2e-3 N·m·s/rad, so a relative band around them is negligible and
`--band` makes it look like "damping barely helps". With
`--bounds "damping=1e-4:0.1"` the optimiser raises damping to ~0.02–0.1 and cuts
the cost substantially.

## Optimiser

`--optimizer cma` (recommended) or `--optimizer de`.

CMA-ES adapts the covariance of its search distribution to the structure of the
cost function, so it handles the correlations and the wide scale spread between
parameters (k ≈ 0.5, d ≈ 1e-3, polynomial coefficients) far better than DE. It
searches a normalised [0,1] box for exactly that reason. At a matched evaluation
budget it wins — clearly so in high-dimensional `per_joint` cases — and it often
terminates early on its own stagnation criterion instead of spending the budget.

![DE vs CMA-ES](../../docs/img/fig_de_vs_cma.png)

`--cma-sigma0` (default 0.25) sets the initial step. Parallel workers ignore
SIGINT so Ctrl+C stays clean.

## Speed

Per-evaluation cost ≈ settling + replay. DE evaluations ≈ popsize·nvars·maxiter
plus polish.

| Lever | Effect |
|---|---|
| `poly` representation | fewer parameters, fewer evaluations |
| `--settling-steps` | the quasi-static settle before **every** replay is ~65 % of per-evaluation time. Lower it for fast exploration, then do a final run at 1000 — it changes the initial equilibrium. |
| `--no-polish` | skips the L-BFGS tail |
| `--workers N` | parallel evaluation (relies on Linux `fork` to inherit the `_GT_*` module globals) |

`frictionloss > 0` adds joint-friction constraints, roughly +8 % per step.

## Typical session

```bash
# 1. baseline
uv run sysid/simulation_based/real2sim.py --mode validate

# 2. explore cheaply
uv run sysid/simulation_based/real2sim.py --mode finetune \
    --optimizer cma --maxiter 100 --settling-steps 200 --no-polish --workers 8 \
    --represent "stiffness=poly:3,damping=poly:3" \
    --bounds "damping=1e-4:0.1,stiffness=0.1:2"

# 3. final run, full settling
uv run sysid/simulation_based/real2sim.py --mode finetune \
    --optimizer cma --maxiter 500 --settling-steps 1000 --workers 8

# 4. bake the result into a model
uv run sysid/simulation_based/apply_params_to_xml.py
```

Outputs land in `build/real2sim/`: `validate.*` and `finetune.{png,json,xml}`.
