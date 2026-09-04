# Models

| File | What it is |
|---|---|
| `spirob_13seg.xml` | the nominal model — 13 joints, 2 tendons, 2 force actuators. The starting point of every identification run. |
| `spirob_13seg_identified.xml` | the same model with the real-to-sim CMA-ES parameters baked in |
| `scene_demo.xml` | rendering-only wrapper: includes the identified model and adds lights, a skybox and a larger offscreen framebuffer. Used by `scripts/render_demo.py`. It changes nothing physical. |

## TL;DR

The body is a **logarithmic spiral**, discretised into rigid segments joined by
hinge joints. Two tendons run along the flanks through site rings and are pulled
by force actuators. Four numbers fix the whole geometry:

| Parameter | Value | Meaning |
|---|---|---|
| `L_target` | 0.44 m | centreline length |
| `base_d` | 0.10 m | diameter at the base |
| `tip_d` | 0.03 m | diameter at the tip |
| `Delta_theta_deg` | 30° | discretisation step per segment |

`SpiralCalculator` solves for the growth parameter `b` by bisection so the
centreline length comes out at `L_target`, then discretises. For the values
above that yields 14 segments / **13 joints**, with segment lengths from
16.9 mm at the base to 51.8 mm at the tip.

```bash
uv run scripts/generate_model.py            # regenerate with the defaults
uv run scripts/generate_model.py --L 0.30 --base-d 0.05 --tip-d 0.01
```

## The tracked model is hand-tuned — this matters

`spirob_13seg.xml` is **not** byte-identical to what `generate_model.py`
produces. It was edited after generation for solver stability during the
identification runs. The differences are exactly these:

| | Generated | Tracked (`spirob_13seg.xml`) |
|---|---|---|
| `timestep` | 0.005 | **0.004** |
| `impratio` | 10 | **15** |
| solver | (default PGS) | **Newton**, `cone="elliptic"` |
| `iterations` | 50 | **20** |
| ground plane `pos` | `0 0 0` | `0 0 -0.053` |
| base body `pos` | `0 0 0.0564484` | `0 0 0` |
| actuator `ctrlrange` | `-50 0` | **`-150 0`** |

The `ctrlrange` change is the important one: the measured tendon forces reach
~110 N, well past the generated −50 N limit, so replaying real data against a
freshly generated model would silently clip the excitation.

**If you regenerate the model, re-apply these edits** — or explicitly decide not
to and re-run `--mode validate` to see what it costs you.

## Joint naming

The XML names count **down** from the base while the MuJoCo index counts **up**:

| model index | XML joint | role |
|---|---|---|
| 0 | `j_12` | base |
| … | … | … |
| 12 | `j_0` | tip |

Every per-joint array in this repository is in **model index order**. See
[`sysid/README.md`](../sysid/README.md#joint-numbering--read-this-before-touching-anything).

## Actuators

```
ctrlrange = [-150, 0]     # pull only -- positive control values are no-ops
data.ctrl[0] = -30.0      # 30 N of tension on tendon 0
```

A sign slip here does not error, it just does nothing — which is why
`tests/test_model.py` asserts the range stays pull-only.
