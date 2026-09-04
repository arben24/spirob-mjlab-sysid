# The MuJoCo model

## TL;DR

13 hinge joints, 2 tendons, 2 **pull-only** force actuators. Joint index 0 is
the base; the XML names count the other way. The tracked model is hand-tuned and
differs from the generator's output in six specific places.

## Structure

```
worldbody
└── base                          (fixed, plus a ground plane)
    └── seg_13   j_12  ← model joint index 0   (base)
        └── seg_12   j_11         index 1
            └── ...
                └── seg_1   j_0   index 12     (tip)

tendon
├── tendon_0     spatial, along one flank
└── tendon_1     spatial, along the other

actuator
├── tendon_act_0   motor on tendon_0, ctrlrange [-150, 0]
└── tendon_act_1   motor on tendon_1, ctrlrange [-150, 0]
```

## Joint index vs. XML name

This trips everyone up exactly once:

| model index | XML joint | body | role |
|---|---|---|---|
| 0 | `j_12` | `seg_13` | **base** |
| 1 | `j_11` | `seg_12` | |
| … | … | … | |
| 12 | `j_0` | `seg_1` | **tip** |

The MuJoCo index follows the kinematic tree *from the base outward*, while the
XML names were assigned counting *down* from the base. So the index goes up as
the name goes down.

Real datasets number `joint_1` = base, so **real `joint_N` → model index
`N−1`**. Every per-joint array in this repository is in model index order.

## Actuators are pull-only

```python
data.ctrl[0] = -30.0   # 30 N of tension on tendon 0
data.ctrl[0] =  30.0   # silently does nothing
```

`ctrlrange = [-150, 0]`. A tendon can pull, never push — as in reality. A sign
error therefore produces a robot that does not move rather than an exception,
which is why `tests/test_model.py` asserts the range explicitly.

The measured tendon forces in `data/trajectories/` peak around **110 N**, which
is why the range extends to −150 rather than the generator's −50.

## Joint parameters

Each joint is a torsion spring with damping and (optionally) dry friction:

| MuJoCo field | Meaning | Identified from |
|---|---|---|
| `jnt_stiffness` | torsional stiffness `k` | free vibration / static load |
| `dof_damping` | viscous damping `d` | free vibration |
| `dof_frictionloss` | dry friction | **nothing** — no measurement exists |
| `dof_armature` | added rotor inertia | optimiser only |
| `jnt_range` | ±24.45° | geometry (segments collide beyond it) |

## Solver settings

```xml
<option timestep="0.004" impratio="15" cone="elliptic" iterations="20"/>
```

Newton solver with an elliptic friction cone. `impratio = 15` biases the solver
toward satisfying friction constraints, which matters once `frictionloss > 0`.
The identification can optimise these as `opt` and `broadcast` groups —
`impratio`, `armature`, `solreflimit_*`, `solimplimit_*` and the friction
variants — because they measurably change the response and no measurement pins
them down.

## Sensors

Registered in `generate_xml_string()` through `SensorRegistry.register()`
**before** `XMLBuilder` runs. Column names in exported data follow a strict
convention:

| Pattern | Example |
|---|---|
| 1-D sensor | `tendon_frc_0`, `tendon_pos_0` |
| 3-D sensor (uppercase axes) | `acc_0_X`, `gyro_0_Z` |
| 3-D geom/estimate (lowercase) | `geom_pos_0_x`, `pos_estimate_0_z` |
| 4-D velocity estimate | `vel_estimate_0_x/y/z/_norm` |
| 4-D quaternion | `quat_estimate_0_w/x/y/z` |

To add a sensor type, register it there and add a matching `DataGroup` in
`spirob/data_schema.py`.

## Hand-tuned deltas

`models/spirob_13seg.xml` is **not** what `scripts/generate_model.py` emits.
Full table in [`models/README.md`](https://github.com/arben24/spirob-mjlab-sysid/blob/main/models/README.md);
the short version is a smaller timestep, `impratio` 15 instead of 10, the Newton
solver, and the wider `ctrlrange`. Regenerating the model without re-applying
these silently changes the identification baseline.
