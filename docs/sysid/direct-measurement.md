# Direct measurement

## TL;DR

Two experiments on isolated joints. The static load test gives stiffness from
Hooke's law; the free-vibration test gives stiffness *and* damping from the
logarithmic decrement of a ring-down. They disagree — by 26 % to 262 % — and the
reasons why are the interesting part.

---

## Joint stiffness: the static load test

The joint is modelled as a **linear torsion spring**, so applied moment and
deflection are proportional:

$$M = k \cdot \Delta\theta$$

The deflection angle `Δθ` is the measured quantity: the torsional stiffness `k`
is the slope of the measurement points in an `M`–`Δθ` diagram.

The moment is produced by applying a force `F` at a defined distance `r` from
the axis of rotation:

$$M = F \cdot r \cdot \sin(\alpha)$$

Here `α` is not the joint angle but the orientation of `F` relative to the lever
arm `r` — a different quantity from `Δθ` entirely. **If the force is applied
perpendicular to the lever, `α = 90°` and this reduces to `M = F·r`.** The setup
is built so that condition holds by construction, which means `α` never has to
be measured and cannot contribute error. Only `F` and `Δθ` remain.

### Evolution of the setup

Several approaches to measuring force and angle were tried and compared.

**Force.** Initially a known mass `m` was hung at a known distance, so
`F = m·g`. Simple, but it needs a separate angle measurement and suffers from
the hanging mass swinging. The better alternative: press the segment onto a
**precision scale** through a cylinder, reading the normal force directly. The
segment presses at a right angle, satisfying `α = 90°`, and the scale proved
more reproducible than hanging masses.

**Angle.** A manual protractor reaches about ±0.5°. Photo-based evaluation —
drawing lines along the outer edges of adjacent segments to the pivot and
measuring the enclosed angle — reaches about ±0.2°.

**The final setup** combines both at maximum precision and repeatability by
using a robot arm: a **Franka Emika Panda** holds one segment while the other is
pressed against the scale at a defined distance.

![Static load setup](../img/setup_static_load_close.jpg)

Force still comes from the scale, because it is more accurate than the arm's own
force estimate (the Panda has no end-effector force sensor and derives force
from its joint torque sensors). The **angle** comes from the arm's joint
encoders: the manufacturer specifies ±0.1 mm end-effector repeatability, and
forward kinematics turns that into a joint angle far more precise than either
alternative.

The arm is driven through ROS. A purpose-built GUI defines a start point, then
drives a sequence of deflection angles. **Each measurement point is the mean of
ten individual measurements**, each a full load cycle: unload, load to the target
angle, read the scale, unload. Averaging full cycles rather than repeated
readings means the scatter *between load cycles* is averaged out too, not just
sensor noise. All points are exported as CSV at the end of a series.

### Evaluation

`F = m·g`, then `M = F·r = m·g·r`. The `(Δθ, M)` pairs are fitted by linear
regression; the slope is `k`.

![Static load evaluation](../img/static_load_joint_01.png)

The points scatter evenly about the regression line with no systematic outliers.

| Joint | `k` [N·m/rad] |
|---|---:|
| 1 (base) | 0.51 |
| 8 (middle) | 0.23 |
| 11 | 0.29 |
| 13 (tip) | 0.27 |

The distribution does **not** show the smooth base-to-tip decrease one would
expect from the tapering geometry. This is attributable to 3D-printing effects.

!!! warning "A known systematic error"
    The weight of the deflected segment itself was not accounted for. Because
    the force is measured along the direction of gravity, the segment's own mass
    enters the scale reading and biases the computed stiffness.

---

## Joint damping: the free vibration test

### The setup, and why it is horizontal

A single segment is clamped so that exactly one joint can swing freely, with the
**joint axis vertical** so the segment swings in a **horizontal plane**. Gravity
therefore acts perpendicular to the direction of motion and parallel to the axis
of rotation. Two things follow:

1. Gravity contributes **no restoring moment**, so the identified stiffness is
   the pure joint stiffness.
2. Gravity has **no tangential component** — it appears only as a constant
   offset on the axis-parallel sensor channel.

Any influence of gravity is excluded *by construction* and never has to be
compensated numerically.

A three-axis accelerometer is embedded in the swinging segment at a known
distance `r_s` from the axis. The rotation produces a tangential acceleration
perpendicular to the radius:

$$a_t = r_s \ddot{\theta}$$

The sensor is oriented so its y-axis coincides with that direction, giving
`θ̈ = a_t / r_s` directly. The centripetal component acts on a different axis and
never enters the evaluation.

### Why the angle is never reconstructed

Recovering `θ(t)` by double integration is deliberately avoided: integrating
noise and sensor offset produces drift that grows with time. Instead the
quantities of interest are taken **straight from the acceleration signal**.

That this is legitimate follows from the model. For the under-damped case
(`d < 2√(kJ)`) the free response is

$$\theta(t) = A e^{-D\omega_0 t}\cos(\omega_d t + \varphi)$$

with `ω₀ = √(k/J)`, damping ratio `D = d/(2√(kJ))` and `ω_d = ω₀√(1−D²)`.

Differentiating twice (collapsing each result back into a single cosine via the
addition theorems and `(Dω₀)² + ω_d² = ω₀²`) gives

$$\ddot{\theta}(t) = A\,\omega_0^2\, e^{-D\omega_0 t}\cos(\omega_d t + \varphi - 2\varphi_v)$$

Compared with the displacement, the acceleration differs **only** by the
constant amplitude factor `ω₀²` and a phase shift. **The envelope `e^{−Dω₀t}`
and the damped frequency `ω_d` are identical.**

Since the logarithmic decrement is the log of the ratio of two successive
maxima one period apart, the constant amplitude factor cancels:

$$\Lambda = \ln\frac{\ddot\theta(t)}{\ddot\theta(t+T_d)} = \ln\frac{A\omega_0^2 e^{-D\omega_0 t}}{A\omega_0^2 e^{-D\omega_0 (t+T_d)}} = D\omega_0 T_d$$

So `Λ` can be taken from the acceleration exactly as from the displacement.

A residual constant sensor offset **is** removed before evaluation, by
subtracting the mean of the decayed tail — an additive shift would corrupt the
amplitude ratio and therefore the decrement.

### From the signal to `k` and `d`

The segment is deflected manually to `θ₀` and released; recording starts at
release. (The signal before the marked start line belongs to the manual
deflection and is ignored.)

![Ring-down](../img/free_vibration_ringdown.png)

From the offset-corrected acceleration:

$$\Lambda = \ln\frac{A_i}{A_{i+1}} = \frac{2\pi D}{\sqrt{1-D^2}}
\qquad\Rightarrow\qquad
D = \frac{\Lambda}{\sqrt{4\pi^2 + \Lambda^2}}$$

Only `ω_d` is directly observable, so `ω₀` is substituted as
`ω₀ = ω_d/√(1−D²)`, leaving both parameters dependent only on `ω_d`, `D` and
`J`:

$$k = \left(\frac{\omega_d}{\sqrt{1-D^2}}\right)^2 J
\qquad
d = \frac{2 D \omega_d}{\sqrt{1-D^2}} J$$

### The moment of inertia is the weak link

`J` is **not** taken from CAD: the mass of a 3D-printed part cannot be derived
accurately from the design because of varying infill density and print
tolerances. Instead the segment is **weighed**, and `J` is estimated from a
simplified geometric approximation of it.

Because `k = ω₀²J`, a relative error in `J` transfers **one-for-one** into the
stiffness. This is the dominant uncertainty in the whole experiment.

### Results

![Free vibration summary](../img/free_vibration_summary.png)

| Joint | `k` [N·m/rad] | `d` [N·m·s/rad] | `D` |
|---|---:|---:|---:|
| 1 (base) | 0.67 | 0.0019 | 0.125 |
| 8 (middle) | 0.83 | 0.0010 | 0.133 |
| 11 | 0.37 | 0.00026 | 0.129 |
| 13 (tip) | 0.69 | 0.0023 | 0.134 |

As with the static test, there is no clean base-to-tip trend — the same printing
effects are responsible.

**Two things validate the method itself:**

1. The logarithmic decrement stays nearly constant across successive maxima and
   the decay is well described by an exponential envelope. Both confirm the
   damping is predominantly **viscous** — a dominant dry-friction component
   would produce a *linear* amplitude decay instead.
2. The damping ratio `D` follows from the decrement alone, is **independent of
   `J`**, and comes out at 0.125–0.133 across all four joints. That consistency
   shows the scatter in `k` comes from the `J` conversion, not from the
   vibration measurement.

### Tooling

![Free vibration GUI](../img/free_vibration_gui.png)

The GUI is where `J` and the per-file start points are set, and it writes the
authoritative `sysid_settings.yaml` per joint. The batch script reproduces the
same analysis but without the manual start points, so it lands a few percent
away — the YAML is the reference.

---

## Comparing the two experiments

| Joint | `k` static | `k` vibration | Deviation |
|---|---:|---:|---:|
| 1 (base) | 0.51 | 0.67 | 31.4 % |
| 8 (middle) | 0.23 | 0.832 | 261.7 % |
| 11 | 0.29 | 0.37 | 27.6 % |
| 13 (tip) | 0.27 | 0.806 | 198.5 % |

*(The thesis numbers are quoted here. The values currently in the repository's
YAML files are 0.670 / 0.832 / 0.370 / 0.692 — joint 13 differs because `J` was
re-tuned afterwards. The point stands either way.)*

The vibration test reports **consistently higher** stiffness. Three
contributions:

1. **Different measurement conditions.** The static test measures a quasi-static
   equilibrium; the vibration test measures a dynamic material response. TPU is
   **viscoelastic**, so its apparent stiffness genuinely varies with frequency.
2. **Measurement error in the static test**, notably the uncompensated weight of
   the deflected segment.
3. **The estimate of `J`.** But this does not explain everything: the 262 %
   discrepancy at joint 8 would require `J` to be off by a factor of 3.6.

## Long-term behaviour

Cyclic tests examined whether stiffness changes over repeated bending, on two
joints with different print structures: the tip-side **joint 13**, which is thin
enough to print as solid TPU, and the base-side **joint 1**, which is thick
enough to be printed with an internal infill structure and its associated air
pockets.

* Joint 13, 100 cycles: moment fell from ~0.8 N·m to ~0.6 N·m (**−25 %**).
* Joint 1, 500 cycles: moment fell from ~0.95 N·m to ~0.8 N·m (**−16 %**).

Both show progressive **material fatigue** with cycle count, independent of print
structure. The relative decrease is *smaller* at joint 1 despite five times the
cycles — but the two joints also differ in geometry and load, so this cannot be
attributed to the solid/infill difference alone. It is a first indication that
long-term behaviour may differ between the two print structures. Whether the
trend continues at higher cycle counts — whether stiffness settles at a level or
keeps falling to failure — needs further tests.

The scatter in these signals is notable and shows the robot arm's integrated
torque sensors are only of limited use for precise stiffness measurement. That
confirms the choice of the external precision scale for the static test.
