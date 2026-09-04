#!/usr/bin/env python3
"""Extract ArUco joint angles from a video and sync them with the force log.

Detects the joint markers (IDs 0-13) and the sync markers (IDs >= 20) in a
video file. The sync markers, which ``record_trajectory.py`` displays on screen
and steps every 5 s, align the video timeline with the CSV recorded from the
motor controller. Everything is merged into a single ``.parquet``.

The output of this script is exactly the format ``real2sim.py`` consumes:
``joint_1_deg`` ... ``joint_13_deg`` (joint 1 = base), ``meas_force_0_N`` /
``meas_force_1_N`` and ``global_timestamp_s``.

Usage::

    uv run sysid/acquisition/track_and_sync.py

Edit the paths at the bottom of the file for your recording. Requires the
``vision`` extra: ``uv pip install -e ".[vision]"``.

Inputs : a video file + build/recordings/recorded_sys_id_auto_data.csv
Outputs: build/recordings/tracked_trajectory.parquet
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from spirob.paths import build_dir as _build_dir


class SyncArucoTracker:
    def __init__(self, video_path: str | Path, csv_path: str | Path, output_path: str | Path):
        self.video_path = Path(video_path)
        self.csv_path = Path(csv_path)
        self.output_path = Path(output_path)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.chain_ids = set(range(14))
        
    def process(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
            
        # ── 1. Read the video and extract markers (Joints + Sync)
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened(): raise ValueError("Could not open video.")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0
        
        print(f"Starte Tracking: {self.video_path.name} | FPS: {fps:.1f} | Frames: {total_frames}")
        
        records = []
        sync_vid_events = {} # marker_id -> list of timestamps
        
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)
            timestamp = frame_idx / fps
            
            if ids is not None:
                seen_markers = set()
                for i, marker_id in enumerate(ids.flatten()):
                    mid = int(marker_id)
                    
                    if mid >= 20:  # Sync Marker
                        if mid not in sync_vid_events:
                            sync_vid_events[mid] = []
                        sync_vid_events[mid].append(timestamp)
                        continue

                    if mid in self.chain_ids and mid not in seen_markers: # Joint Marker
                        seen_markers.add(mid)
                        c = corners[i][0]
                        dx = float(c[1, 0] - c[0, 0] + c[2, 0] - c[3, 0])
                        dy = float(c[1, 1] - c[0, 1] + c[2, 1] - c[3, 1])
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        cx = float(c[:, 0].mean())
                        cy = float(c[:, 1].mean())
                        
                        records.append({
                            "frame": frame_idx,
                            "video_timestamp_s": timestamp,
                            "marker_id": int(marker_id),
                            "x": cx,
                            "y": cy,
                            "angle_deg": angle
                        })
            
            frame_idx += 1
            if frame_idx % 250 == 0:
                elapsed = time.time() - start_time
                print(f"Verarbeitet: {frame_idx}/{total_frames} ({frame_idx/elapsed:.1f} FPS)")
                
        cap.release()
        
        if not records:
            print("No joint markers found.")
            return

        # ── 2. Create Video DataFrame
        df_long = pl.DataFrame(records, schema={
            "frame": pl.UInt32, "video_timestamp_s": pl.Float64,
            "marker_id": pl.Int32, "x": pl.Float64, "y": pl.Float64, "angle_deg": pl.Float64
        })
        
        df_vid = df_long.pivot(on="marker_id", index=["frame", "video_timestamp_s"], values=["x", "y", "angle_deg"])
        
        # Fill missing frames
        df_frames = pl.DataFrame({"frame": range(frame_idx)}, schema={"frame": pl.UInt32})
        df_vid = df_frames.join(df_vid, on="frame", how="left").with_columns(
            pl.col("video_timestamp_s").fill_null(pl.col("frame") / fps)
        )

        # Interpolate markers
        marker_cols = [c for c in df_vid.columns if c not in ["frame", "video_timestamp_s"]]
        if marker_cols:
            df_vid = df_vid.with_columns([pl.col(c).interpolate().forward_fill().backward_fill() for c in marker_cols])

        # Calculate joint angles
        joint_exprs = []
        for i in range(1, 14):
            prev_ang, curr_ang = f"angle_deg_{i-1}", f"angle_deg_{i}"
            if prev_ang in df_vid.columns and curr_ang in df_vid.columns:
                joint_ang = (pl.col(curr_ang) - pl.col(prev_ang) + 180) % 360 - 180
                joint_exprs.append(joint_ang.alias(f"joint_{i}_deg"))
        if joint_exprs:
            df_vid = df_vid.with_columns(joint_exprs)

        # ── 3. Synchronize with CSV
        df_csv = pl.read_csv(self.csv_path)

        # Find median offset between video and CSV
        # We will take the FIRST time a marker appears in video and the FIRST time it appears in CSV.
        offsets = []
        csv_sync_events = df_csv.group_by("aruco_id").agg(pl.col("global_timestamp_s").min())
        csv_sync_dict = {row["aruco_id"]: row["global_timestamp_s"] for row in csv_sync_events.to_dicts()}

        for mid, vid_times in sync_vid_events.items():
            if mid in csv_sync_dict:
                # offset: how much to add to video_time to get csv_time
                offset = csv_sync_dict[mid] - vid_times[0] 
                offsets.append(offset)
                
        if not offsets:
            print("WARNING: no matching sync markers found between video and CSV.")
            offset_val = 0.0
        else:
            offset_val = float(np.median(offsets))
            print(f"Sync succeeded. Time offset (CSV - video): {offset_val:.3f} s")

        # Map video time to CSV time
        df_vid = df_vid.with_columns((pl.col("video_timestamp_s") + offset_val).alias("global_timestamp_s"))

        # ── 4. Merge Video and Sensor Data
        # We interpolate the CSV sensor data onto the video timestamps
        # Convert to numpy for interpolation
        csv_time = df_csv["global_timestamp_s"].to_numpy()
        vid_time = df_vid["global_timestamp_s"].to_numpy()
        
        interp_cols = ["cmd_force_0_N", "cmd_force_1_N", "meas_force_0_N", "meas_force_1_N", 
                       "meas_length_0_mm", "meas_length_1_mm"]
        
        merged_cols = []
        for col in interp_cols:
            if col in df_csv.columns:
                csv_vals = df_csv[col].to_numpy()
                # Use linear interpolation, holding boundaries
                f_interp = interp1d(csv_time, csv_vals, kind='linear', bounds_error=False, fill_value=(csv_vals[0], csv_vals[-1]))
                interp_vals = f_interp(vid_time)
                merged_cols.append(pl.Series(col, interp_vals))
                
        df_vid = df_vid.with_columns(merged_cols)

        # ── 5. Save combined Parquet
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_vid.write_parquet(self.output_path)
        
        print("\nMerge completed successfully.")
        print(f"Data saved to: {self.output_path}")
        print(df_vid.head())

if __name__ == "__main__":
    base_dir = _build_dir("recordings")
    # Point these at your own recording
    video_file = base_dir / "Video" / "GX010070.MP4" 
    csv_file = base_dir / "recorded_sys_id_auto_data.csv"
    output_file = base_dir / "tracked_trajectory.parquet"
    
    tracker = SyncArucoTracker(video_file, csv_file, output_file)
    try:
        tracker.process()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
