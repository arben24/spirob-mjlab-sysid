"""The shipped measurement data must stay in the shape the scripts expect.

Every identification result in the docs is reproduced from these files, so a
renamed column or a missing YAML key silently invalidates the documentation.
"""

import csv

import polars as pl
import pytest
import yaml

from spirob.paths import (
    DEFAULT_PARAMS,
    DEFAULT_TRAJECTORY,
    FREE_VIBRATION_DIR,
    STATIC_LOAD_DIR,
)

ANCHOR_JOINTS = ["joint_01", "joint_08", "joint_11", "joint_13"]


@pytest.mark.parametrize("joint", ANCHOR_JOINTS)
def test_free_vibration_folder_is_complete(joint):
    folder = FREE_VIBRATION_DIR / joint
    assert folder.is_dir(), f"{folder} missing"
    assert list(folder.glob("spirob_messung_*.csv")), "no ring-down recordings"

    doc = yaml.safe_load((folder / "sysid_settings.yaml").read_text())
    # J is hand-tuned and k/d scale linearly with it -- it must be present.
    assert "J" in doc["settings"]
    for key in ("k_mean", "d_mean", "zeta_mean", "n_valid"):
        assert key in doc["results"], f"{joint}: results.{key} missing"


def test_damping_ratio_is_consistent_across_joints():
    """zeta follows from the log decrement alone and is independent of J, so it
    should come out near-constant even where the stiffnesses scatter."""
    zetas = [
        yaml.safe_load((FREE_VIBRATION_DIR / j / "sysid_settings.yaml").read_text())
        ["results"]["zeta_mean"]
        for j in ANCHOR_JOINTS
    ]
    assert all(0.10 < z < 0.16 for z in zetas), zetas


@pytest.mark.parametrize("joint", ANCHOR_JOINTS)
def test_static_load_series_parses(joint):
    with open(STATIC_LOAD_DIR / f"{joint}.csv") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.lstrip().startswith("#")))
    assert len(rows) >= 6
    assert {"mass_g", "angle_deg"} <= set(rows[0])


def test_reference_trajectory_has_the_expected_columns():
    df = pl.read_parquet(DEFAULT_TRAJECTORY)
    required = {"global_timestamp_s", "meas_force_0_N", "meas_force_1_N"}
    required |= {f"joint_{i}_deg" for i in range(1, 14)}
    assert required <= set(df.columns), sorted(required - set(df.columns))
    assert len(df) > 1000


def test_identified_parameter_set_covers_every_joint():
    import json

    doc = json.loads(DEFAULT_PARAMS.read_text())
    assert len(doc["joints"]) == 13
    # Index 0 must be the base joint; the whole real<->sim mapping hangs on it.
    assert doc["joints"][0]["model_joint"] == "j_12"
    assert doc["joints"][0]["real_joint"] == "joint_1"
