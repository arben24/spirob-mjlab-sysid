"""The generated model and the tracked model XMLs must both stay loadable.

These are guard-rails, not physics checks: if the generator or a hand edit
breaks the kinematic chain, the tendon routing or the actuator limits, every
identification script downstream fails in a much more confusing way.
"""

import mujoco as mj
import numpy as np
import pytest

import spirob
from spirob.paths import DEFAULT_MODEL, IDENTIFIED_MODEL, MODELS_DIR, RL_MODEL


def test_generated_model_matches_the_nominal_geometry():
    geom = spirob.SpiralCalculator(
        L_target=0.44, base_d=0.10, tip_d=0.03, Delta_theta_deg=30.0
    ).compute_geometry()
    # The bisection has to hit the requested centreline length.
    assert np.isclose(sum(geom.seg_lengths), 0.44, atol=1e-6)


def test_generated_xml_loads_and_has_the_expected_topology():
    xml = spirob.generate_xml_string(0.44, 0.10, 0.03, 30.0, "Spirob")
    model = mj.MjModel.from_xml_string(xml)
    assert model.njnt == 13
    assert model.ntendon == 2
    assert model.nu == 2


@pytest.mark.parametrize(
    "path", [DEFAULT_MODEL, IDENTIFIED_MODEL, RL_MODEL, MODELS_DIR / "scene_demo.xml"]
)
def test_tracked_models_load(path):
    model = mj.MjModel.from_xml_path(str(path))
    assert model.njnt == 13
    assert model.ntendon == 2


def test_actuators_are_pull_only():
    """Positive control is a no-op by design; a sign slip here would silently
    push the tendons and make every identification meaningless."""
    model = mj.MjModel.from_xml_path(str(DEFAULT_MODEL))
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        assert lo < 0 and hi == 0, f"actuator {i} ctrlrange {lo}..{hi} is not pull-only"


def test_model_steps_without_diverging():
    model = mj.MjModel.from_xml_path(str(IDENTIFIED_MODEL))
    data = mj.MjData(model)
    data.ctrl[:] = -20.0
    for _ in range(500):
        mj.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.max(np.abs(data.qpos)) < 1e3


def test_rl_model_keeps_what_the_tasks_assume():
    """`rl/` trains against `RL_MODEL`, and the task configs hard-code parts of
    it: two tendons named `tendon_0`/`tendon_1`, a `site_tcp`, one `site_imu_i`
    per segment, and the pull-only ctrlrange the action term maps onto. mjlab is
    not installed in this environment, so this is the only place those
    assumptions get checked in CI."""
    model = mj.MjModel.from_xml_path(str(RL_MODEL))

    names = lambda objtype, n: {  # noqa: E731
        mj.mj_id2name(model, objtype, i) for i in range(n)
    }
    assert names(mj.mjtObj.mjOBJ_TENDON, model.ntendon) == {"tendon_0", "tendon_1"}
    site_names = names(mj.mjtObj.mjOBJ_SITE, model.nsite)
    assert "site_tcp" in site_names
    assert {f"site_imu_{i}" for i in range(14)} <= site_names

    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        assert (lo, hi) == (-150.0, 0.0), "TendonEffortActionCfg maps onto [-150, 0]"

    # The task's TENDON_REST_LEN subtracts the straight-pose tendon length from
    # the observation. If the model changes, that constant has to change too.
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert np.allclose(data.ten_length, 0.4258, atol=1e-3), (
        "mdp/constants.py TENDON_REST_LEN no longer matches the model"
    )
