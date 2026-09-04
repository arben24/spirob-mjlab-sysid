# CLAUDE.md

Guidance for Claude Code (claude.ai/code) and other AI agents working in this
repository.

## What this repository is

MuJoCo model generation and experimental system identification for the SpiRob, a
tendon-driven quasi-continuum robot with a logarithmic-spiral body. The
identification half is **complete**; the reinforcement-learning half
([`rl/`](rl/)) is **scaffolded but not implemented**.

Read [`README.md`](README.md) first, then the folder README of wherever you are
working. They are the specification; this file is the operating manual.

## Commands

Everything uses **uv**. Never pip, conda or manual virtualenvs.

```bash
uv venv && uv pip install -e ".[vision,hardware,gui,docs,dev]"

uv run pytest                                    # 32 tests, no hardware
uv run ruff check .

uv run scripts/generate_model.py                 # model from spiral parameters
uv run scripts/render_demo.py --gif              # headless video (EGL)

uv run sysid/direct/static_load.py               # static load test, ~2 s
uv run sysid/direct/free_vibration.py            # ring-down batch, ~1 min
uv run sysid/direct/free_vibration_gui.py <dir>  # interactive; needs a display

uv run sysid/simulation_based/real2sim.py --mode validate     # ~1 min
uv run sysid/simulation_based/real2sim.py --mode finetune --optimizer cma --maxiter 500 --workers 8
uv run sysid/simulation_based/sim2sim.py --compare

uv run sysid/figures/fig_validation.py           # ~1 min (re-simulates 60 s)
uv run sysid/figures/fig_sim2sim.py              # ~5 s

uv run mkdocs serve                              # documentation site
```

## Non-negotiables

**Paths.** Never hardcode a path. Import from `spirob.paths`: `DATA_DIR`,
`MODELS_DIR`, `BUILD_DIR`, `DEFAULT_MODEL`, `IDENTIFIED_MODEL`,
`DEFAULT_TRAJECTORY`, `DEFAULT_PARAMS`, and `build_dir("subdir")` for output.
Every script reads from `data/` + `models/` and writes to `build/`. `build/` is
git-ignored; nothing generated is ever tracked.

**Language.** The repository is English — code, comments, docstrings, console
output, figure labels, documentation. The one deliberate exception is
`typst_table()` in `sim2sim.py`, which emits a German table for the author's
thesis. Figures switch to German decimal commas with `SPIROB_FIG_LOCALE=de`;
that is the *only* localisation mechanism, and it lives in
`spirob.plotstyle` (`num`, `sci`, `auto`, `localize_axes`). The `de_*` and
`german_axes` names are back-compat aliases — do not use them in new code.

**Joint order.** One convention, no exceptions. Model index `i` = MuJoCo joint
`i`; the tree runs from the base, so index 0 = base = `j_12` and index 12 = tip
= `j_0`. The XML names count down while the indices count up. Real datasets
number `joint_1` = base, so real `joint_N` → model index `N−1`. Every per-joint
array — seeds, `qpos`, stiffness/damping/frictionloss vectors, optimiser output
— is in model index order. `joint_index_labels()` in `real2sim.py` is the single
source of truth. `--reverse-real-joints` exists only for a dataset numbered from
the tip; the shipped data is not.

**Actuators are pull-only.** `ctrlrange = [-150, 0]`. A positive control value
is a silent no-op, not an error. `tests/test_model.py` guards this.

**`spirob_13seg.xml` is hand-tuned**, not byte-identical to the generator's
output. The deltas are documented in [`models/README.md`](models/README.md).
If you regenerate it, re-apply them or explicitly decide not to — the
`ctrlrange` one in particular, because measured forces reach ~110 N.

**Keep `sysid/` and `rl/` separate.** `sysid/` produces a model, `rl/` consumes
one. No imports across that line; both may use `src/spirob/`.

## The result you must not misrepresent

The identification does **not** produce a model that matches reality. The three
methods (static load, free vibration, trajectory fit) disagree, and the fitted
parameters are physically implausible. This is the finding, not a bug:

> The rigid-chain + torsion-spring model has no term for tendon friction in the
> guide rings, TPU non-linearity or joint coupling. The optimiser compensates
> with parameter combinations that minimise the cost function and mean nothing
> physically. The sim-to-real gap here is a **modelling** problem, not an
> **optimisation** problem.

Sim-to-sim works fine (tendon stiffness recovered to <5 %), which is what proves
the optimiser is not at fault. If you find yourself "fixing" the identification
to get better numbers, you are probably fitting noise — read
[`docs/sysid/results.md`](docs/sysid/results.md) first.

## Architecture

```
four spiral parameters (L, base_d, tip_d, Δθ)
    → SpiralCalculator (bisection for b)  → SpiralGeometry
    → XMLBuilder + SensorRegistry         → MJCF XML string
    → mj.MjModel.from_xml_string()

measured tendon forces + ArUco joint angles (data/trajectories/*.parquet)
    → load_and_preprocess()   moving-average smoothing, outlier rejection
    → simulate_and_sample()   quasi-static settle, then force replay
    → compute_cost()          weighted RMSE(pos) + 0.05 · RMSE(vel), in degrees
    → CMA-ES / DE             → build/real2sim/finetune.{json,xml,png}
```

| File | Role |
|---|---|
| `src/spirob/spiral.py` | pure logarithmic-spiral maths, no MuJoCo |
| `src/spirob/generator.py` | `SpiralCalculator`, `XMLBuilder`, `SensorRegistry` |
| `src/spirob/simulate.py` | rollout loop, controllers, contact forces, video |
| `src/spirob/paths.py` | every canonical path in the repo |
| `src/spirob/plotstyle.py` | shared figure style + locale switch |
| `sysid/simulation_based/real2sim.py` | the main identification (1900 lines) |
| `sysid/direct/free_vibration_gui.py` | where `J` and the per-joint `k`/`d` are set |

Import from `spirob` (the package `__init__`), not from submodules, except
`spirob.paths` and `spirob.plotstyle` which are imported directly by convention.

## Gotchas

* **`J` is hand-tuned, not measured.** `k = ω₀²J` and `d ∝ J`, so both scale
  linearly with a guess. It lives in `settings.J` in each joint's
  `sysid_settings.yaml`. The damping ratio `ζ` is independent of `J` and comes
  out consistently at 0.13 ± 0.01 — that consistency is the sanity check.
* **The GUI YAML is authoritative**, not the batch script. `free_vibration.py`
  does not apply the per-file manual start points, so it differs by a few
  percent by design.
* **`frictionloss` has no measurement.** Uniform seed 0.15, wide band.
* **Damping seeds are ~1e-3**, so a relative `--band` around them is
  meaningless. Use absolute `--bounds "damping=1e-4:0.1"`.
* **Tendon seeds come from the XML** (stiffness 50). An old hardcoded 500 was
  far too stiff; removing it dropped the validate cost ~7×.
* **`--settling-steps` is ~65 % of per-evaluation time.** Lower it to explore,
  then do a final run at 1000 — it changes the initial equilibrium.
* **`differential_evolution(workers>1)` relies on Linux `fork`** to inherit the
  `_GT_*` module globals. It will not work on spawn-based platforms.
* **Unstable sims return a finite `1e6`**, never NaN, so the optimiser is
  steered away instead of corrupted.
* **Ctrl+C during a finetune** stops cleanly and still saves the best result.
* **EGL teardown prints a harmless `EGLError`** at interpreter exit after
  headless rendering. Ignore it.
* **`.gitignore` blanket-ignores images and PDFs** with a negation for
  `docs/img/**`. To publish a new figure, copy it from `build/` into `docs/img/`.

## Documentation

Markdown throughout, built with MkDocs Material, deployed to GitHub Pages by
`.github/workflows/docs.yml` on push to `main`.

* `README.md` — the overview and the headline results. Keep it short.
* `docs/` — the long-form write-up (`model/`, `sysid/`, `rl/`).
* Folder `README.md` files — how to *use* that folder.

`docs/index.md` is generated from `README.md` during the CI build; do not edit
it by hand. Every document opens with a **TL;DR** section — keep that pattern.

When you change a script's behaviour, update the folder README in the same
commit. When you change a *finding*, update `docs/sysid/results.md` and the
README table together.
