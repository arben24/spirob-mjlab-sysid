# System identification

## TL;DR

The central problem in applying reinforcement learning to soft robotics is the
**sim-to-real gap**: a policy trained in simulation must keep working on real
hardware that differs in dynamics, sensing and manufacturing tolerances. How
well that transfer works depends directly on how faithfully the simulation
reproduces the physical robot.

For the SpiRob this is hard. It is a tendon-driven quasi-continuum robot made of
elastic material: compliant, non-linear, and dominated by friction effects —
above all the tendon rubbing against the guide rings of each segment, and the
transitions between static and sliding friction. The simplified MuJoCo model
represents it as a chain of rigid segments with torsion-spring joints; those
friction effects are not in it at all.

The joint stiffness and damping cannot be read off a drawing or a datasheet.
They have to be measured. This chapter is how.

**Two approaches were used:**

| | [Direct measurement](direct-measurement.md) | [Simulation-based](simulation-based.md) |
|---|---|---|
| Starts from | the physical part | the observed behaviour |
| Method | isolate one joint, measure it | replay real forces in MuJoCo, optimise until the motion matches |
| Yields | physically interpretable `k`, `d` | whatever minimises the cost |
| Verdict | methodologically sound, but the values disagree between the two experiments and do not reproduce real motion in the simulator | reproduces gross motion only; the parameters are physically implausible |

**And the conclusion:** [the results page](results.md) — the gap is a
**modelling problem, not an optimisation problem.**

## Why the parameters must be measured at all

The model was already structurally in place from a preceding research project,
but its joint parameters had only been *estimated* for lack of measurement data.
Every measurement setup, identification method and evaluation described here was
built for this work.

## Why one joint at a time

A joint cannot be characterised on the assembled robot: deflecting one segment
always moves its neighbours, so nothing is isolated. Each measurement therefore
clamps one of the two connected segments and moves the other.

The robot has **13 joints**, and because the logarithmic spiral tapers
continuously from base to tip, the mechanical parameters should vary along its
length. To capture that without measuring all 13, **four representative joints**
were measured: **1 (base), 8 (middle), 11 and 13 (near the tip)** — spanning the
full length. All other joints are linearly interpolated between the nearest
measured ones, which is a good approximation because the segment size ratio is
close to one, so the parameter profile is flat enough between the support
points.

## Why two different experiments

Stiffness and damping are fundamentally different physical phenomena. Stiffness
is a **static restoring moment**; damping is **velocity-dependent energy
dissipation**. One experiment cannot capture both:

* **Joint stiffness** — a *static load test* gives the torsional stiffness
  directly from Hooke's law. Simple to run and precise, but limited to static
  equilibrium and says nothing about damping.
* **Joint damping** — damping requires a dynamic measurement, so a *free
  vibration test* is used. A segment carrying an accelerometer is deflected and
  released; assuming a linear equation of motion, the decay of the resulting
  damped oscillation yields **both** the stiffness and the damping.

That overlap is useful: the two experiments both report stiffness, and comparing
them is what exposes how uncertain these numbers really are.
