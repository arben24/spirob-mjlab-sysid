# Geometry

## TL;DR

Four numbers in, a segment chain out. The only non-trivial step is solving for
the spiral's growth parameter `b`, which is done by bisection because the
centreline length has no closed-form inverse.

## The spiral

A logarithmic spiral in polar form:

$$\rho(\theta) = a \cdot e^{b\theta}$$

`a` sets the scale, `b` the growth rate. The ratio between successive
discretisation steps is constant:

$$\beta = \frac{\rho_{i+1}}{\rho_i} = e^{b \cdot \Delta\theta}$$

which is exactly the self-similarity that makes the shape useful for a
continuum robot.

## Solving for `b`

The user specifies the *centreline length* `L_target` and the base and tip
diameters. Those over-determine `b`, so it has to be solved for:

1. `θ₀` follows from the diameter ratio: `theta0_from_ratio(b, base_d, tip_d)`.
2. `a` follows from the tip diameter: `a_from_tip(b, tip_d)`.
3. The centreline length `L(b)` is then a closed-form integral —
   `length_central(a, b, θ₀)`.
4. `f(b) = L(b) − L_target` is driven to zero by `scipy.optimize.bisect`.

For very small `|b|` the length formula is poorly conditioned; the module
documents this and the working range stays well away from it.

All of this lives in `spirob.spiral` as **pure functions with no global state**,
which is why it is the one part of the repository with real unit-test coverage
(`tests/test_spiral.py`).

## Discretisation

With `b`, `a` and `θ₀` known, the spiral is walked in steps of `Δθ`. Each step
produces one segment: a centreline length and a half-width at each end. The
effective step is adjusted slightly (29.45° rather than 30°) so a whole number
of segments fits `θ₀` exactly.

`SpiralCalculator.compute_geometry()` returns a `SpiralGeometry` with
`seg_lengths`, the half-widths and a `summary()` for printing.

```python
import spirob

geom = spirob.SpiralCalculator(
    L_target=0.44, base_d=0.10, tip_d=0.03, Delta_theta_deg=30.0
).compute_geometry()
print(geom.summary())
```

## From geometry to MJCF

`XMLBuilder` turns the segment list into MJCF: one `<body>` per segment with a
box geom and a hinge joint, sites on both flanks for the tendon to run through,
two `<spatial>` tendons, two force actuators, and the sensors registered by
`SensorRegistry`.

```python
xml = spirob.generate_xml_string(0.44, 0.10, 0.03, 30.0, "Spirob")
```

!!! warning "Units"
    `generate_xml_string` takes `Delta_theta_deg` in **degrees**. Passing
    radians produces a model with thousands of segments that MuJoCo rejects with
    `XML_ELEMENT_DEPTH_EXCEEDED` — a confusing error for a unit slip.
