"""
Simple Segment Position Estimator using ACC and GYRO data.

This module provides a basic dead-reckoning position estimation for robot segments
based on accelerometer and gyroscope measurements.
"""

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation as R


class SegmentState(NamedTuple):
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    orientation: np.ndarray  # quaternion [w, x, y, z]

class SimpleSegmentEstimator:
    """
    Dead-reckoning position estimator for robot segments using ACC + GYRO.
    
    Integrates gyroscope for orientation, transforms ACC to world frame,
    subtracts gravity, and double-integrates for position.
    """
    
    def __init__(self, 
                 segment_ids: list[int],
                 initial_positions: dict[int, list[float]] | None = None,
                 initial_orientations: dict[int, list[float]] | None = None):
        """
        Initialize estimator for multiple segments.
        
        Args:
            segment_ids: List of segment IDs to track
            initial_positions: Dict of {seg_id: [x, y, z]} initial positions
            initial_orientations: Dict of {seg_id: [w, x, y, z]} initial quaternions
        """
        self.segment_ids = segment_ids
        self.states: dict[int, SegmentState] = {}
        
        for seg_id in segment_ids:
            pos = np.array(initial_positions.get(seg_id, [0.0, 0.0, 0.0]))
            quat = np.array(initial_orientations.get(seg_id, [1.0, 0.0, 0.0, 0.0]))
            vel = np.zeros(3)
            
            self.states[seg_id] = SegmentState(position=pos, velocity=vel, orientation=quat)
    
    def update_batch(self, sensor_data: dict[int, dict[str, list[float]]], dt: float):
        """
        Update all segments with a batch of sensor data.
        
        Args:
            sensor_data: {seg_id: {'acc': [ax, ay, az], 'gyro': [gx, gy, gz]}}
            dt: Time step in seconds
        """
        for seg_id, data in sensor_data.items():
            if seg_id not in self.states:
                continue
                
            acc = np.array(data['acc'])
            gyro = np.array(data['gyro'])
            
            self._update_segment(seg_id, acc, gyro, dt)
    
    def _update_segment(self, seg_id: int, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Update a single segment."""
        state = self.states[seg_id]
        
        # Update orientation using gyroscope (simple integration)
        rot = R.from_quat(state.orientation)
        rot = rot * R.from_rotvec(gyro * dt)
        new_quat = rot.as_quat()
        
        # Transform acceleration to world frame
        world_acc = rot.apply(acc)
        
        # Subtract gravity (assuming z-up)
        world_acc[2] -= 9.81
        
        # Update velocity and position
        new_vel = state.velocity + world_acc * dt
        new_pos = state.position + new_vel * dt
        
        # Update state
        self.states[seg_id] = SegmentState(
            position=new_pos,
            velocity=new_vel,
            orientation=new_quat
        )
    
    def get_state(self, seg_id: int) -> SegmentState:
        """Get current state of a segment."""
        return self.states[seg_id]
    
    def get_all_states(self) -> dict[int, SegmentState]:
        """Get states of all segments."""
        return self.states.copy()