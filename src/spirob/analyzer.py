import json
from pathlib import Path

import polars as pl

from .data_schema import ExperimentRecord


def load_experiment(run_id: str, base_dir: str = "build") -> tuple[ExperimentRecord, pl.LazyFrame]:
    """
    Loads an experiment by validating metadata and opening a LazyFrame to the Parquet data.

    Args:
        run_id: Unique run identifier.
        base_dir: Base directory for experiments.

    Returns:
        Tuple of (ExperimentRecord, LazyFrame)
    """
    base_path = Path(base_dir) / "experiments" / run_id
    meta_path = base_path / "meta.json"
    data_path = base_path / "data.parquet"

    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"Experiment {run_id} not found in {base_path}")

    # Load and validate metadata
    with open(meta_path) as f:
        meta_dict = json.load(f)
    try:
        record = ExperimentRecord(**meta_dict)
    except Exception as e:
        print(f"Warning: Failed to validate meta.json for {run_id}: {e}")
        raise FileNotFoundError(f"Invalid meta.json for {run_id}")

    # Open LazyFrame
    lf = pl.scan_parquet(str(data_path))

    return record, lf

def example_analysis(record: ExperimentRecord, lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Example analysis: Compute mean values for each sensor group dynamically.

    Args:
        record: ExperimentRecord with metadata.
        lf: LazyFrame to the data.

    Returns:
        DataFrame with mean values.
    """
    results = []
    for sensor in record.sensors:
        if sensor.dimension == 1:
            mean_val = lf.select(pl.col(sensor.columns[0]).mean()).collect().item()
            results.append({"sensor": sensor.name, "group": sensor.group.value, "mean": mean_val})
        elif sensor.dimension == 3:
            means = lf.select([pl.col(col).mean() for col in sensor.columns]).collect()
            mean_x, mean_y, mean_z = means[sensor.columns[0]].item(), means[sensor.columns[1]].item(), means[sensor.columns[2]].item()
            results.append({"sensor": sensor.name, "group": sensor.group.value, "mean_x": mean_x, "mean_y": mean_y, "mean_z": mean_z})

    return pl.DataFrame(results)