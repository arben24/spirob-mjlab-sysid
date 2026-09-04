import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataGroup(Enum):
    ACC = "accelerometer"
    GYRO = "gyroscope"
    TENDON_FRC = "tendon_force"
    TENDON_POS = "tendon_position"
    TENDON_VEL = "tendon_velocity"
    JOINT_POS = "joint_position"
    JOINT_VEL = "joint_velocity"
    GEOM_POS = "geom_position"
    FORCE_LOCAL = "force_local"
    FORCE_GLOBAL = "force_global"
    BODY_CONTACT_FRC = "body_contact_force"

class SensorMeta(BaseModel):
    name: str = Field(..., description="Name of the sensor or geom")
    group: DataGroup = Field(..., description="Group category")
    dimension: int = Field(..., description="Number of dimensions (1, 3, etc.)")
    unit: str = Field(..., description="Unit of measurement, e.g., 'm/s^2'")
    columns: list[str] = Field(..., description="List of column names in the DataFrame")

class ExperimentConfig(BaseModel):
    L_target: float = Field(..., description="Target length")
    base_d: float = Field(..., description="Base diameter")
    tip_d: float = Field(..., description="Tip diameter")
    Delta_theta_deg: float = Field(..., description="Delta theta in degrees")
    sim_time: float = Field(..., description="Simulation time in seconds")
    controller_info: str | None = Field(default=None, description="Controller description")
    geom_type: str = Field(..., description="Geometry type, e.g., 'Cylinder'")
    geom_params: dict[str, Any] = Field(..., description="Geometry parameters")
    include_geom_pos: bool = Field(default=False, description="Whether geom positions are included")
    enable_position_estimation: bool = Field(default=False, description="Enable position estimation from ACC+GYRO")
    position_estimator_segments: list[int] | None = Field(default=None, description="Specific segments to estimate, None for all")
    initial_positions: dict[int, list[float]] | None = Field(default=None, description="Initial positions {seg_id: [x,y,z]}")
    initial_orientations: dict[int, list[float]] | None = Field(default=None, description="Initial orientations {seg_id: [w,x,y,z]}")

class ExperimentRecord(BaseModel):
    run_id: str = Field(..., description="Unique run identifier")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Creation timestamp")
    config: ExperimentConfig | None = Field(default=None, description="Simulation configuration")
    sensors: list[SensorMeta] = Field(..., description="List of sensor metadata")
    version: str = Field(default="1.0", description="Schema version")