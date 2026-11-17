"""Immutable prediction-time what-if transformations over Phase 3 inputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from aiqon.prediction.features import build_features
from aiqon.schemas.shipment import Shipment
from aiqon.schemas.snapshots import CheckpointSnapshot
from aiqon.dashboard.services.prediction_adapter import Phase3Prediction, Phase3PredictionAdapter


class ScenarioParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    port_congestion: float | None = Field(default=None, ge=0, le=1)
    weather_severity: float | None = Field(default=None, ge=0, le=1)
    berth_capacity: int | None = Field(default=None, ge=1, le=10)
    customs_workload: float | None = Field(default=None, ge=0, le=1)
    berth_service_hours: float | None = Field(default=None, gt=0, le=48)
    inland_traffic: float | None = Field(default=None, ge=0, le=1)


@dataclass(frozen=True)
class ScenarioResult:
    label: str
    original: Phase3Prediction
    modified: Phase3Prediction
    parameters: dict
    affected_parameters: dict[str, str]
    stage_deltas_hours: dict[str, float]
    remaining_change_hours: float
    eta_change_hours: float
    probability_change: float
    interval_lower_change_hours: float | None
    interval_upper_change_hours: float | None
    source_snapshot_sha256_before: str
    source_snapshot_sha256_after: str
    source_state_unchanged: bool


class ScenarioService:
    """Run synthetic assumption changes without simulator or persistence calls."""
    def __init__(self, predictor: Phase3PredictionAdapter):
        self.predictor = predictor

    @staticmethod
    def _snapshot_digest(snapshot: CheckpointSnapshot) -> str:
        payload = json.dumps(snapshot.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def parameter_support(self, snapshot: CheckpointSnapshot, shipment: Shipment,
                          route=None, world=None) -> dict[str, tuple[bool, str]]:
        route = route or self.predictor.routes.get(shipment.route_id)
        world = world or self.predictor.bundle["world"]
        if route is None:
            raise ValueError(f"Unknown route {shipment.route_id!r}.")
        cp = snapshot.checkpoint_type
        features = build_features(snapshot, shipment, route, world)
        active_port = cp in {"departure", "port_arrival"}
        has_weather = (features.get("observed_port_weather") is not None
                       or features.get("forecast_port_weather") is not None)
        return {
            "port_congestion": (active_port,
                "Port congestion can affect the remaining port-handling estimate." if active_port else
                "Port handling is complete at customs clearance."),
            "weather_severity": (active_port and has_weather,
                "An eligible checkpoint-visible weather feature is available; propagation does not consume weather, and the trained residual model may be insensitive at this shipment." if active_port and has_weather else
                "No eligible port-weather observation/forecast is available at this checkpoint." if active_port else
                "Port weather is historical by customs clearance and is not a future operating input."),
            "berth_capacity": (active_port, "Berth operations are complete at customs clearance."),
            "berth_service_hours": (active_port, "Berth service is complete at customs clearance."),
            "customs_workload": (cp != "customs_cleared", "Customs processing is complete at this checkpoint."),
            "inland_traffic": (True, "Inland traffic uses an eligible forecast, current observation, or configured route fallback."),
        }

    def run(self, snapshot: CheckpointSnapshot, shipment: Shipment,
            parameters: ScenarioParameters | dict) -> ScenarioResult:
        params = parameters if isinstance(parameters, ScenarioParameters) else ScenarioParameters.model_validate(parameters)
        route = self.predictor.routes.get(shipment.route_id)
        if route is None:
            raise ValueError(f"Unknown route {shipment.route_id!r}.")
        world = self.predictor.bundle["world"]
        support = self.parameter_support(snapshot, shipment, route, world)
        selected_parameters = params.model_dump(exclude_none=True)
        unsupported = [name for name in selected_parameters if not support[name][0]]
        if unsupported:
            explanations = "; ".join(f"{name}: {support[name][1]}" for name in unsupported)
            raise ValueError(f"Scenario parameter is unsupported at {snapshot.checkpoint_type}: {explanations}")

        before = self._snapshot_digest(snapshot)
        original = self.predictor.predict(snapshot, shipment)
        modified_snapshot = snapshot.model_copy(deep=True)
        modified_route = route.model_copy(deep=True)
        modified_world = world.model_copy(deep=True)
        env = dict(modified_snapshot.current_environment)
        forecasts = list(modified_snapshot.forecasts)

        feature_to_forecast = {
            "port_congestion": "port_congestion", "weather_severity": "port_weather",
            "customs_workload": "customs_workload", "inland_traffic": "inland_traffic",
        }
        for parameter, forecast_variable in feature_to_forecast.items():
            if parameter not in selected_parameters:
                continue
            value = float(selected_parameters[parameter])
            env_key = "port_weather" if parameter == "weather_severity" else forecast_variable
            if env_key in env:
                env[env_key] = value
            forecasts = [forecast.model_copy(update={"forecast_value": value})
                         if forecast.forecast_variable == forecast_variable else forecast
                         for forecast in forecasts]
            if parameter == "port_congestion":
                modified_route = modified_route.model_copy(update={
                    "congestion": modified_route.congestion.model_copy(update={"baseline": value})})
            elif parameter == "customs_workload":
                modified_route = modified_route.model_copy(update={
                    "customs": modified_route.customs.model_copy(update={"workload_baseline": value})})
            elif parameter == "inland_traffic":
                modified_route = modified_route.model_copy(update={
                    "inland": modified_route.inland.model_copy(update={"traffic_baseline": value})})

        if "berth_capacity" in selected_parameters:
            modified_route = modified_route.model_copy(update={"berth_capacity": selected_parameters["berth_capacity"]})
        if "berth_service_hours" in selected_parameters:
            modified_world = modified_world.model_copy(update={
                "berth_service_base_hours": float(selected_parameters["berth_service_hours"])})
        modified_snapshot = modified_snapshot.model_copy(update={"current_environment": env,
                                                                  "forecasts": tuple(forecasts)})
        modified = self.predictor.predict(modified_snapshot, shipment, route=modified_route, world=modified_world)
        after = self._snapshot_digest(snapshot)
        if before != after:
            raise RuntimeError("What-if scenario unexpectedly mutated the source snapshot.")
        stage_deltas = {stage: modified.propagation_components.get(stage, 0.0)
                        - original.propagation_components.get(stage, 0.0)
                        for stage in sorted(set(original.propagation_components) | set(modified.propagation_components))}
        low_delta = (None if original.remaining_hours_lower is None or modified.remaining_hours_lower is None
                     else modified.remaining_hours_lower - original.remaining_hours_lower)
        high_delta = (None if original.remaining_hours_upper is None or modified.remaining_hours_upper is None
                      else modified.remaining_hours_upper - original.remaining_hours_upper)
        return ScenarioResult(
            label="SIMULATED SCENARIO — NOT AN ACTUAL SHIPMENT OUTCOME",
            original=original, modified=modified, parameters=selected_parameters,
            affected_parameters={name: support[name][1] for name in selected_parameters},
            stage_deltas_hours=stage_deltas,
            remaining_change_hours=modified.predicted_remaining_hours - original.predicted_remaining_hours,
            eta_change_hours=(modified.predicted_delivery_at - original.predicted_delivery_at).total_seconds() / 3600.0,
            probability_change=modified.probability_delay_over_12h - original.probability_delay_over_12h,
            interval_lower_change_hours=low_delta, interval_upper_change_hours=high_delta,
            source_snapshot_sha256_before=before, source_snapshot_sha256_after=after,
            source_state_unchanged=before == after)

    @staticmethod
    def reset_parameters() -> ScenarioParameters:
        """Reset means inherit the selected snapshot/configuration unchanged."""
        return ScenarioParameters()

