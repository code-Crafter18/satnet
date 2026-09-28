from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware and UTC.")
    return value.astimezone(timezone.utc)


class RiskLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SatelliteTLE(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1, max_length=128)
    norad_id: int = Field(gt=0)
    line1: str = Field(min_length=69, max_length=69)
    line2: str = Field(min_length=69, max_length=69)
    epoch: datetime
    source: str = "unknown"
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validation_status: str = "valid"

    @field_validator("epoch", "ingested_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class PropagationConfig(BaseModel):
    start_time: datetime
    end_time: datetime
    time_step_seconds: int = Field(60, ge=1, le=86400)

    @field_validator("start_time", "end_time")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("end_time")
    @classmethod
    def range_ok(cls, value: datetime, info):
        start = info.data.get("start_time")
        if start and value <= start:
            raise ValueError("end_time must be after start_time")
        return value


class StateVector(BaseModel):
    timestamp: datetime
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
    frame: str = "TEME"
    position_unit: str = "km"
    velocity_unit: str = "km/s"

    @field_validator("timestamp")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class Trajectory(BaseModel):
    satellite: SatelliteTLE
    states: list[StateVector]
    frame: str = "TEME"
    position_unit: str = "km"
    velocity_unit: str = "km/s"


class ConjunctionEvent(BaseModel):
    satellite_a: str
    satellite_b: str
    tca: datetime
    miss_distance_km: float = Field(ge=0)
    relative_velocity_km_s: float = Field(ge=0)
    pc: float | None = Field(default=None, ge=0, le=1)
    pc_status: str = "unavailable"
    risk_level: RiskLevel
    risk_reason: str
    # Hybrid risk fields
    ml_pc: float | None = Field(default=None, description="ML-predicted collision probability")
    confidence: str | None = Field(default=None, description="HIGH / MEDIUM / LOW")
    fusion_method: str | None = Field(default=None, description="consensus / ml_only / physics_only / geometric_only")
    needs_review: bool = Field(default=False, description="Flagged for human review when physics and ML disagree")


class SimulationSummary(BaseModel):
    simulation_id: str
    status: str
    satellite_count: int
    candidate_pairs: int
    conjunction_count: int
    duration_seconds: float
    processing_time_seconds: float
    ml_enabled: bool
    ml_fallback_used: bool
    warning: str | None = None
    conjunctions: list[ConjunctionEvent] = Field(default_factory=list)
