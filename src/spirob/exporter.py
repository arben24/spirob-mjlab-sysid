import json
from pathlib import Path

import polars as pl

from .data_schema import DataGroup, ExperimentRecord, SensorMeta


def generate_sensor_meta(df: pl.DataFrame) -> list[SensorMeta]:
    """
    Generates SensorMeta list from DataFrame columns.
    Assumes column naming convention: group_name_X/Y/Z for 3D, group_name for 1D.
    """
    sensors = []
    columns = df.columns
    group_map = {
        "acc": (DataGroup.ACC, "m/s^2"),
        "gyro": (DataGroup.GYRO, "rad/s"),
        "tendonfrc": (DataGroup.TENDON_FRC, "N"),
        "tendonpos": (DataGroup.TENDON_POS, "m"),
        "tendonvel": (DataGroup.TENDON_VEL, "m/s"),
        "jointpos": (DataGroup.JOINT_POS, "rad"),
        "jointvel": (DataGroup.JOINT_VEL, "rad/s"),
        "geompos": (DataGroup.GEOM_POS, "m"),
        "bodycontactfrc": (DataGroup.BODY_CONTACT_FRC, "N"),
    }

    for col in columns:
        if col == "time_s":
            continue
        # Special handling for body contact forces
        if "_contact_force_" in col and col.endswith(("_X", "_Y", "_Z")):
            base_name = col[:-2]
            if base_name + "_X" in columns and base_name + "_Y" in columns and base_name + "_Z" in columns:
                name = base_name
                if not any(s.name == name and s.group == DataGroup.BODY_CONTACT_FRC for s in sensors):
                    sensors.append(SensorMeta(
                        name=name,
                        group=DataGroup.BODY_CONTACT_FRC,
                        dimension=3,
                        unit="N",
                        columns=[base_name + "_X", base_name + "_Y", base_name + "_Z"]
                    ))
            continue  # Skip further processing for this column
        parts = col.split("_")
        if len(parts) >= 2:
            group_prefix = parts[0]
            if group_prefix in group_map:
                group, unit = group_map[group_prefix]
                name = "_".join(parts[0:-1]) if len(parts) > 2 else "_".join([parts[0], parts[1]])  # naming takes place here
                if col.endswith("_X") or col.endswith("_Y") or col.endswith("_Z"):
                    # 3D sensor
                    base_name = col[:-2]  # remove _X
                    if base_name + "_X" in columns and base_name + "_Y" in columns and base_name + "_Z" in columns:
                        if not any(s.name == name and s.group == group for s in sensors):
                            sensors.append(SensorMeta(
                                name=name,
                                group=group,
                                dimension=3,
                                unit=unit,
                                columns=[base_name + "_X", base_name + "_Y", base_name + "_Z"]
                            ))
                else:
                    # 1D sensor
                    if not any(s.name == name and s.group == group for s in sensors):
                        sensors.append(SensorMeta(
                            name=name,
                            group=group,
                            dimension=1,
                            unit=unit,
                            columns=[col]
                        ))
    return sensors

def save_experiment(df: pl.DataFrame, record: ExperimentRecord, base_dir: str = "build") -> str:
    """
    Saves the experiment data and metadata.

    Args:
        df: Polars DataFrame with time series data.
        record: ExperimentRecord with metadata.
        base_dir: Base directory for experiments.

    Returns:
        Path to the experiment folder.
    """
    base_path = Path(base_dir) / "experiments" / record.run_id
    base_path.mkdir(parents=True, exist_ok=True)

    # Save data as Parquet
    data_path = base_path / "data.parquet"
    df.write_parquet(str(data_path))

    # Validate that body contact force columns are present (optional)
    body_force_cols = [col for col in df.columns if "_contact_force_" in col]
    if not body_force_cols:
        print(f"Warning: No body contact force columns found in {data_path}. "
              "Body force metadata will be skipped. "
              "Use include_body_forces=True in simulation if needed.")
        # Continue without raising error

    # Save metadata as JSON
    meta_path = base_path / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump(record.model_dump(mode='json'), f, indent=4)

    return str(base_path)