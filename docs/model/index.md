# The model

## TL;DR

The SpiRob is a 3D-printed, tendon-driven quasi-continuum robot. Its body is a
**logarithmic spiral**, tapering from a 100 mm base to a 30 mm tip over a
440 mm centreline. In MuJoCo it is modelled as a chain of **rigid segments
joined by hinge joints**, where each joint is a **torsion spring with damping**.
Two tendons run along the flanks and are pulled by force actuators.

That abstraction is the whole story of this repository — including its
limitations. See [Geometry](geometry.md) for how the spiral becomes segments,
and [MuJoCo model](mujoco.md) for what the MJCF actually contains.

<p align="center">
  <img src="../img/spirob_demo.gif" alt="SpiRob curling" width="600">
</p>

## Why a logarithmic spiral

A logarithmic spiral has a constant ratio between successive turns. Applied to a
tapered continuum body this gives a **constant size ratio between neighbouring
segments**, which in turn means:

* the curvature the robot can reach grows smoothly toward the tip,
* the shape it curls into is self-similar at every scale,
* and the mechanical parameters should vary *gradually* along the chain rather
  than jumping from joint to joint.

That last point is what makes it defensible to measure only four joints and
interpolate the rest, and what motivates the polynomial parameter
representation in the identification.

## Why a rigid chain

MuJoCo has no native continuum model. A chain of rigid bodies with compliant
joints is the standard discretisation: it is fast, stable, and differentiable
enough for RL. The price is that everything genuinely continuum-like — the
distributed compliance of the TPU, the tendon sliding through its guide rings,
the coupling between neighbouring joints — has to be squeezed into per-joint
stiffness, damping and friction.

The system identification in this repository is an attempt to find the best
possible values under that abstraction. Its main finding is that
**no values are good enough**, because the missing terms are structural. See
[the results](../sysid/results.md).

## Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `L_target` | 0.44 m | centreline length |
| `base_d` | 0.10 m | diameter at the base |
| `tip_d` | 0.03 m | diameter at the tip |
| `Delta_theta_deg` | 30° | discretisation step per segment |

which yields:

| Derived | Value |
|---|---|
| growth parameter `b` | 0.167314 |
| scale factor `β` = ρᵢ₊₁/ρᵢ | 1.089804 |
| scale factor `a` | 0.016118 m |
| total angle `θ₀` | 7.196 rad (412.3°) |
| taper angle `φ` | 9.096° |
| segments | 14 (→ **13 joints**) |
| segment length | 16.9 mm (base) … 51.8 mm (tip) |

```bash
uv run scripts/generate_model.py
```
