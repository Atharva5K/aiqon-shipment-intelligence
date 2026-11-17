"""Load and validate project YAML configuration."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

EXPECTED_CHECKPOINTS = ("departure", "port_arrival", "customs_cleared")


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def timezone_must_be_known(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"timezone must be a valid IANA timezone name, got {value!r}") from exc
        return value


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_shipments: int = Field(gt=0)
    random_seed: int
    checkpoint_names: tuple[str, ...]

    @field_validator("checkpoint_names")
    @classmethod
    def check_checkpoint_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != EXPECTED_CHECKPOINTS:
            raise ValueError(f"checkpoint_names must be exactly {EXPECTED_CHECKPOINTS}")
        return value


class PredictionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    significant_delay_threshold_hours: float = Field(gt=0)
    conformal_coverage: float = Field(gt=0, lt=1)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_splits: int = Field(ge=2)
    n_repeats: int = Field(ge=1)
    bootstrap_samples: int = Field(ge=1)
    primary_metric: Literal["mae"]
    phase3: "Phase3EvaluationConfig"


class Phase3ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(ge=1)
    min_samples_leaf: int = Field(ge=1)
    random_seed: int


class Phase3EvaluationConfig(BaseModel):
    """Reproducible Phase 3 evaluation and sensitivity experiment settings."""

    model_config = ConfigDict(extra="forbid")
    fit_fraction: float = Field(gt=0, lt=1)
    calibration_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    split_seed: int
    model: Phase3ModelConfig
    counterfactual_timestamp_tolerance_seconds: float = Field(gt=0)
    minimum_scale_hours: float = Field(gt=0)
    probability_threshold: float = Field(ge=0, le=1)
    random_mask_probability: float = Field(ge=0, lt=1)
    local_explanation_count: int = Field(ge=1)
    stress_transit_delay_hours: float = Field(gt=0)
    stress_peak_congestion: float = Field(ge=0, le=1)
    stress_customs_processing_multiplier: float = Field(gt=1)
    stress_gate_cutoff_hour_utc: int = Field(ge=0, le=23)
    stress_reduced_berth_capacity: int = Field(ge=1)
    stress_shipment_count: int = Field(ge=3)
    stress_weekend_start_at_utc: datetime
    stress_weekend_scheduled_transit_hours: float = Field(gt=0)
    stress_weekend_transit_perturbation_hours: float = Field(gt=0)
    stress_weekend_injected_transit_delay_hours: float = Field(gt=0)
    missingness_scenarios: dict[str, tuple[str, ...]]
    attribution_reference_values: dict[str, float]
    stress_seed_base: int

    @field_validator("stress_weekend_start_at_utc")
    @classmethod
    def stress_start_is_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("stress_weekend_start_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def split_fractions_sum_to_one(self):
        if abs(self.fit_fraction + self.calibration_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("Phase 3 fit, calibration and test fractions must sum to 1")
        if self.fit_fraction != 0.7 or self.calibration_fraction != 0.15 or self.test_fraction != 0.15:
            raise ValueError("Phase 3 requires the fixed 70/15/15 shipment split")
        return self


class RoutesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_path: str = Field(min_length=1)


class WorldConfig(BaseModel):
    """Global Phase 1 assumptions, separate from unchanged route profiles."""

    model_config = ConfigDict(extra="forbid")
    start_at_utc: datetime
    departure_window_days: int = Field(gt=0)
    transit_perturbation_hours: float = Field(gt=0)
    berth_service_base_hours: float = Field(gt=0)

    @field_validator("start_at_utc")
    @classmethod
    def start_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("start_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig
    data: DataConfig
    prediction: PredictionConfig
    evaluation: EvaluationConfig
    routes: RoutesConfig
    world: WorldConfig


class CongestionConfig(BaseModel):
    """Normalized synthetic congestion values; baseline and volatility are in [0, 1]."""

    model_config = ConfigDict(extra="forbid")
    baseline: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)


class CustomsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operating_days: tuple[Literal["monday", "tuesday", "wednesday", "thursday",
                                 "friday", "saturday", "sunday"], ...]
    opening_hour_utc: int = Field(ge=0, le=23)
    closing_hour_utc: int = Field(ge=1, le=24)
    base_processing_hours: float = Field(gt=0)
    workload_baseline: float = Field(default=0.50, ge=0, le=1,
                                     description="Fallback customs-workload indicator in [0, 1]; matches synthetic environment initialization.")

    @field_validator("operating_days")
    @classmethod
    def operating_days_are_nonempty_and_unique(cls, value):
        if not value:
            raise ValueError("operating_days must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("operating_days must be unique")
        return value

    @model_validator(mode="after")
    def opening_precedes_closing(self):
        if self.opening_hour_utc >= self.closing_hour_utc:
            raise ValueError("opening_hour_utc must be earlier than closing_hour_utc")
        return self


class InlandConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    travel_hours: float = Field(gt=0)
    gate_cutoff_hour_utc: int = Field(ge=0, le=23)
    traffic_baseline: float = Field(default=0.45, ge=0, le=1,
                                    description="Fallback inland-traffic indicator in [0, 1]; matches synthetic environment initialization.")


class RouteProfile(BaseModel):
    """Typed documented synthetic corridor parameters; no simulation behavior."""

    model_config = ConfigDict(extra="forbid")
    route_id: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    destination_port: str = Field(min_length=1)
    final_destination: str = Field(min_length=1)
    scheduled_transit_hours: float = Field(gt=0)
    berth_capacity: int = Field(ge=1)
    berth_slot_hours: tuple[int, ...] = Field(min_length=1)
    congestion: CongestionConfig
    customs: CustomsConfig
    inland: InlandConfig

    @field_validator("berth_slot_hours")
    @classmethod
    def berth_slots_are_valid_and_unique(cls, value):
        if any(hour < 0 or hour > 23 for hour in value):
            raise ValueError("berth_slot_hours values must be between 0 and 23 inclusive")
        if len(value) != len(set(value)):
            raise ValueError("berth_slot_hours values must be unique")
        return value


class RouteFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routes: tuple[RouteProfile, ...]

    @field_validator("routes")
    @classmethod
    def check_route_ids(cls, value: tuple[RouteProfile, ...]) -> tuple[RouteProfile, ...]:
        ids = [route.route_id for route in value]
        if len(ids) != len(set(ids)):
            raise ValueError("routes must have unique route_id values")
        if len(value) != 3:
            raise ValueError("the initial route configuration must contain exactly 3 routes")
        return value


def _read_yaml(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Configuration root in {path} must be a YAML mapping")
    return result


def load_config(path: str | Path) -> SimulationConfig:
    """Load the main configuration and validate its values."""
    config_path = Path(path)
    try:
        return SimulationConfig.model_validate(_read_yaml(config_path))
    except ValidationError as exc:
        raise ValueError(f"Invalid simulation configuration in {config_path}: {exc}") from exc


def load_routes(path: str | Path) -> RouteFile:
    """Load and validate the three initial synthetic route profiles."""
    route_path = Path(path)
    try:
        return RouteFile.model_validate(_read_yaml(route_path))
    except ValidationError as exc:
        raise ValueError(f"Invalid route configuration in {route_path}: {exc}") from exc
