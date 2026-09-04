# Figures

Publication-quality figures from the identification results.

```bash
uv run sysid/figures/fig_validation.py    # the real-to-sim validation figure
uv run sysid/figures/fig_sim2sim.py       # optimiser comparison, sensitivity, identifiability
```

Both write PDF (vector, for documents) and PNG (for quick viewing) into
`build/figures/`.

## TL;DR

| Script | Produces | Reads |
|---|---|---|
| `fig_validation.py` | `fig_real2sim_validation` | a `real2sim.py` result JSON — **re-simulates** the trajectory so the figure shows exactly what the cost function scored |
| `fig_sim2sim.py` | `fig_de_vs_cma`, `fig_sensitivity`, `fig_identifiability` | `sim2sim.py` result JSONs |

`fig_sim2sim.py` falls back to the shipped results in `data/identified/` when no
fresh run exists, so the figures reproduce from a clean clone.

## Design rules

All figures use `spirob.plotstyle`, which enforces three things:

1. **Colour-vision deficiency** — the categorical slots come from a palette
   checked against deuteranopia, protanopia and tritanopia; all ten pairings of
   the five slots hold their OKLab separation.
2. **Greyscale print** — identity never rests on colour alone. Each slot also
   carries its own dash pattern, marker and hatch. Secondary encoding is used
   only where several series share one panel, never as decoration.
3. **Quiet chrome** — thin spines, fine grid, text in ink rather than in the
   series colour.

The 13 joints are an **ordered** quantity, so they get a single-hue sequential
ramp with a colour bar, not 13 categorical colours.

## Language

Figures render in English by default. For the German thesis variants (decimal
comma):

```bash
SPIROB_FIG_LOCALE=de uv run sysid/figures/fig_validation.py
```
