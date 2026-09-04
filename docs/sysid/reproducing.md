# Reproducing the results

## TL;DR

Everything in this documentation reproduces from a clean clone with no hardware.
The raw measurements are in `data/`; every script writes to `build/`.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                                                  #  ~30 s
uv run sysid/direct/static_load.py                             #   ~2 s
uv run sysid/direct/free_vibration.py                          #  ~60 s
uv run sysid/simulation_based/real2sim.py --mode validate      #  ~60 s
uv run sysid/figures/fig_validation.py                         #  ~60 s
uv run sysid/figures/fig_sim2sim.py                            #   ~5 s
```

Only the finetune stage is expensive (hours), and its result is shipped in
`data/identified/` so every figure reproduces without re-running it.

## What each command reproduces

| Command | Reproduces |
|---|---|
| `sysid/direct/static_load.py` | the static-load stiffness table and per-joint figures |
| `sysid/direct/free_vibration.py` | the ring-down `k`, `d`, `ζ` per joint and the summary figure |
| `sysid/direct/free_vibration_gui.py <dir> --figure` | the GUI figure and the ring-down trace |
| `real2sim.py --mode validate` | the baseline cost with the measured seeds |
| `fig_validation.py` | the real-to-sim validation figure — **re-simulates** the full 60 s |
| `fig_sim2sim.py` | the optimiser comparison, sensitivity and identifiability figures |
| `scripts/render_demo.py --gif` | the animation on the landing page |

## Re-running the expensive fit

```bash
uv run sysid/simulation_based/real2sim.py --mode finetune \
    --optimizer cma --maxiter 500 --workers 8
```

Expect hours on 8 cores. Ctrl+C stops cleanly and still writes the best result
so far. Output: `build/real2sim/finetune.{json,png,xml}`.

To explore cheaply first, cut the settling steps and skip the polish — both
change the result, so do a final run at the real settings:

```bash
uv run sysid/simulation_based/real2sim.py --mode finetune \
    --optimizer cma --maxiter 100 --settling-steps 200 --no-polish --workers 8 \
    --represent "stiffness=poly:3,damping=poly:3" \
    --bounds "damping=1e-4:0.1,stiffness=0.1:2"
```

## Re-running the sim-to-sim comparison

```bash
uv run sysid/simulation_based/sim2sim.py --compare --workers 10 --tol 1e-8
uv run sysid/simulation_based/sim2sim.py --sensitivity --workers 10
uv run sysid/figures/fig_sim2sim.py
```

`fig_sim2sim.py` prefers a fresh run in `build/sim2sim/` and falls back to the
shipped results in `data/identified/` otherwise.

## Figures for the thesis

The figures render in English by default. For the German variants with decimal
commas:

```bash
SPIROB_FIG_LOCALE=de uv run sysid/figures/fig_validation.py
```

Both PDF (vector, for the document) and PNG (for quick viewing) are written.

## Reproducing the raw data

That needs the hardware — the ESP32 motor controller, the sensor boards, the
Franka Emika Panda and a camera. See
[`sysid/acquisition/README.md`](https://github.com/arben24/spirob-mjlab-sysid/blob/main/sysid/acquisition/README.md)
for the serial protocol, the ArUco setup and the recording chain.

## Publishing a new figure

`.gitignore` blanket-ignores images, with a negation for `docs/img/**`. So:

```bash
uv run sysid/figures/fig_validation.py
cp build/figures/fig_real2sim_validation.png docs/img/
```

Nothing in `build/` is ever tracked.
