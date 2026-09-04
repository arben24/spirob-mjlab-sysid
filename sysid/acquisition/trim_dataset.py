import sys
from pathlib import Path

import polars as pl

from spirob.paths import build_dir as _build_dir


class DataTrimmer:
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
    def trim_by_time(self, start_time_s: float = 0.0, end_time_s: float = None, reset_zero: bool = True):
        """
        Trim the dataset by global time (global_timestamp_s).
        Optionally resets the counters (time, frames) back to 0.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
            
        print(f"Reading data from: {self.input_path}")
        df = pl.read_parquet(self.input_path)
        
        original_len = len(df)
        
        # Base minima, in case we want to stay relative to the original start
        original_min_global = df["global_timestamp_s"].min()
        
        # 1. Filter the data
        filter_expr = pl.lit(True)
        if start_time_s > 0:
            actual_start = original_min_global + start_time_s
            filter_expr = filter_expr & (pl.col("global_timestamp_s") >= actual_start)
            
        if end_time_s is not None:
            actual_end = original_min_global + end_time_s
            filter_expr = filter_expr & (pl.col("global_timestamp_s") <= actual_end)
            
        df = df.filter(filter_expr)
            
        new_len = len(df)
        print(f"Datensatz getrimmt: {original_len} -> {new_len} Zeilen")
        
        if new_len == 0:
            print("WARNING: the resulting dataset is empty. Check the trim values.")
            return None
            
        # 2. Reset time and frame counters to 0
        if reset_zero:
            print("Resetting global time, video time and frame IDs to 0 ...")
            df = df.with_columns([
                (pl.col("global_timestamp_s") - pl.col("global_timestamp_s").min()).alias("global_timestamp_s"),
                (pl.col("video_timestamp_s") - pl.col("video_timestamp_s").min()).alias("video_timestamp_s"),
                (pl.col("frame") - pl.col("frame").min()).alias("frame")
            ])
        else:
            print("Time axes and frame IDs stay absolute (no reset to 0).")
        
        # 3. Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.output_path)
        print(f"Saved to: {self.output_path}")

        return df

if __name__ == "__main__":
    base_dir = _build_dir("recordings")
    input_file = base_dir / "tracked_trajectory.parquet"
    output_file = base_dir / "tracked_trajectory_trimmed.parquet"
    
    trimmer = DataTrimmer(input_file, output_file)
    
    # Trim configuration. Both bounds are relative to the START of the
    # original dataset.
    # (0 = the beginning). END_TIME_SECONDS = None does not trim the tail.
    START_ZEIT_SEKUNDEN = 10.0 
    END_ZEIT_SEKUNDEN = 35.0
    
    try:
        df_result = trimmer.trim_by_time(start_time_s=START_ZEIT_SEKUNDEN, end_time_s=END_ZEIT_SEKUNDEN, reset_zero=True)
        if df_result is not None:
            print("\nPreview of the trimmed data:")
            print(df_result.select(["frame", "global_timestamp_s", "video_timestamp_s"]).head())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
