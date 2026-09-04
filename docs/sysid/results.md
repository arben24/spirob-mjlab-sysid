# Results and discussion

## TL;DR

Two fundamentally different approaches, both pursuing the same goal from
opposite directions. Both fall short, and they fall short for the *same* reason:

> **The current simulation model is structurally unsuited to describing the real
> dynamics of the SpiRob. As long as essential physical influences are not
> represented in the model, no identification method can deliver parameters that
> are simultaneously physically plausible and behaviourally accurate. The
> sim-to-real gap here is not primarily an optimisation problem — it is a
> modelling problem.**

The practical answer is a two-stage strategy: calibrate as far as the model
structure allows, then bridge the remainder with **domain randomisation**.

---

## Comparing the two methods

The **direct measurement** starts from the physical system and tries to capture
its mechanical properties by measuring them. It delivers **physically
interpretable** values for torsional stiffness and damping coefficient — both
derivable from material properties and joint geometry, and therefore
methodologically well founded.

Nevertheless the static load test and the free vibration test partly disagree
(26 % to 262 %), and no consistent profile emerges across the joints.
Manufacturing effects and the methods' own measurement uncertainties are
responsible. And when the values are transferred into the MuJoCo model, a clear
discrepancy appears between simulated and real motion.

The reason is that the simulation model is limited to a simplified
torsion-spring-damper representation, and therefore systematically omits real
phenomena — the tendon friction in the guide channels, the non-linear behaviour
of TPU under large deformation. **The directly measured parameters describe the
joint under idealised conditions, not the complete dynamic system of the
SpiRob.**

The **simulation-based identification** attacks exactly that point: it does not
derive parameters physically but back-computes them from the observed overall
behaviour.

In the sim-to-sim scenario the method's basic suitability is confirmed — the
optimiser finds parameters that reconstruct synthetically generated trajectories
with small errors. In the real-to-sim scenario the actual difficulty emerges. The
optimisation converges and exhausts the scope the model allows, but reaches a
markedly poorer match with the real reference trajectory. The identified
parameters deviate substantially from physically plausible magnitudes and
fluctuate between neighbouring joints with no discernible physical relationship.
**The optimiser compensates for the missing model terms with parameter
combinations that are numerically favourable but physically uninterpretable.**

For that reason the concrete numerical values of the identified parameters carry
no independent explanatory value. They are recorded in
[`data/identified/`](https://github.com/arben24/spirob-mjlab-sysid/tree/main/data/identified)
purely for traceability of the training configuration.

## The numbers

**Stiffness, the two direct experiments**

| Joint | `k` static | `k` vibration | Deviation |
|---|---:|---:|---:|
| 1 (base) | 0.51 | 0.67 | 31.4 % |
| 8 (middle) | 0.23 | 0.832 | 261.7 % |
| 11 | 0.29 | 0.37 | 27.6 % |
| 13 (tip) | 0.27 | 0.806 | 198.5 % |

**Identifiability from trajectory data (sim-to-sim, the best case)**

| Quantity | Error range | Verdict |
|---|---|---|
| tendon stiffness | < 5 % | reliably identifiable |
| joint damping | 4 – 31 % | usable |
| joint stiffness | 28 – 158 % | **not reliably identifiable** |

**Real-to-sim (CMA-ES, 500 iterations)**

| Metric | Value |
|---|---|
| cost `J` | 9.857 |
| mean joint RMSE | 6.8° |
| best / worst joint | 3.0° / 14.2° |

## What is missing from the model

| Real effect | In the model? |
|---|---|
| torsional joint stiffness | yes |
| viscous joint damping | yes |
| dry joint friction | approximated by `frictionloss`, never measured |
| **tendon friction in the guide rings** | **no** |
| **stick–slip transitions** | **no** |
| **TPU non-linearity under large deformation** | **no** |
| **coupling between neighbouring joints** | **no** |
| **actuation hysteresis, rope winding dynamics** | **no** |
| material fatigue over cycles | no (measured to be ~25 % over 100 cycles) |

That such effects *can* be captured when the model structure is extended
accordingly is shown for the same robot type by work in which the motor and
rope-winding dynamics are part of the system model alongside the compliant body,
thereby reproducing actuation hysteresis and self-contact at the motion limits.

## The consequence: a two-stage strategy

For a successful transfer of simulation-learned policies to real hardware, this
implies two stages:

1. **Calibrate.** Use the parameter identification described here to bring the
   simulation model as close to the real SpiRob's actual motion patterns as the
   current model structure permits.
2. **Randomise.** Because structural modelling gaps cannot be closed this way, a
   systematic residual uncertainty remains between simulation and reality. Bridge
   it with **domain randomisation**: train the agent not on a single, exactly
   calibrated model but on many randomised model variants covering the range of
   genuinely possible but not exactly known system parameters.

Under domain randomisation the individual numerical value loses further
significance, since the basis of training is a whole parameter *range* rather
than one exactly calibrated set. That is precisely why the implausibility of the
identified parameters is tolerable — and why it must not be papered over.

See [`rl/README.md`](https://github.com/arben24/spirob-mjlab-sysid/blob/main/rl/README.md)
for suggested randomisation ranges derived from the error figures above.
