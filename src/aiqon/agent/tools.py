"""Strict, read-only tool registry backed by existing Phase 3/4 services."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aiqon.agent.config import AgentConfigurationError
from aiqon.dashboard.services.dashboard_service import DashboardService
from aiqon.dashboard.services.explanation_service import ExplanationBundle
from aiqon.dashboard.services.scenario_service import ScenarioParameters, ScenarioService

Checkpoint = Literal["departure", "port_arrival", "customs_cleared"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class ShipmentArgs(_StrictModel):
    shipment_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class EtaArgs(ShipmentArgs):
    checkpoint: Checkpoint | None = None


class QueryFilters(_StrictModel):
    checkpoint: Checkpoint | None = None
    predicted_delay_above_hours: float | None = Field(default=None, ge=-8760, le=8760)
    significant_delay_probability_above: float | None = Field(default=None, ge=0, le=1)
    route_id: str | None = Field(default=None, min_length=1, max_length=80)
    cargo_type: str | None = Field(default=None, min_length=1, max_length=80)
    shipment_ids: list[str] | None = Field(default=None, min_length=1, max_length=25)
    limit: int = Field(default=10, ge=1, le=25)

    @classmethod
    def model_validate(cls, obj, **kwargs):
        # Preserve pydantic's strict JSON parsing while bounding every ID.
        result = super().model_validate(obj, **kwargs)
        if result.shipment_ids and any(not value or len(value) > 64 for value in result.shipment_ids):
            raise ValueError("shipment_ids must contain nonempty IDs of at most 64 characters")
        return result


class QueryArgs(_StrictModel):
    filters: QueryFilters


class WhatIfArgs(ShipmentArgs):
    checkpoint: Checkpoint | None = None
    scenario_parameters: ScenarioParameters

    @model_validator(mode="after")
    def require_non_null_scenario_parameter(self):
        if not any(value is not None for value in self.scenario_parameters.model_dump().values()):
            raise ValueError("scenario_parameters must include at least one non-null supported parameter")
        return self


def _iso_utc(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone.utc)
    else:
        stamp = stamp.tz_convert(timezone.utc)
    return stamp.isoformat().replace("+00:00", "Z")


def _jsonable(value):
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        return _iso_utc(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prediction_dict(prediction) -> dict:
    return {
        "shipment_id": prediction.shipment_id,
        "checkpoint": prediction.checkpoint_type,
        "prediction_timestamp_utc": _iso_utc(prediction.prediction_created_at),
        "predicted_remaining_hours": float(prediction.predicted_remaining_hours),
        "predicted_delivery_at_utc": _iso_utc(prediction.predicted_delivery_at),
        "predicted_total_delay_hours": float(prediction.predicted_total_delay_hours),
        "significant_delay_probability": float(prediction.probability_delay_over_12h),
        "probability_status": prediction.probability_status,
        "remaining_time_interval_hours": None if prediction.remaining_hours_lower is None else {
            "lower": float(prediction.remaining_hours_lower),
            "upper": float(prediction.remaining_hours_upper),
        },
        "interval_status": prediction.interval_status,
        "propagation_remaining_hours": float(prediction.propagation_remaining_hours),
        "ml_residual_correction_hours": float(prediction.ml_residual_hours),
        "propagation_components_hours": _jsonable(prediction.propagation_components),
        "model_version": prediction.model_version,
        "feature_schema_version": prediction.feature_schema_version,
    }


class AgentToolService:
    """Deterministic tool implementations; never reads dashboard outcome joins."""

    def __init__(self, dashboard: DashboardService, explanations: ExplanationBundle,
                 scenarios: ScenarioService, *, selected_shipment_id: str,
                 selected_checkpoint: str):
        self.dashboard = dashboard
        self.explanations = explanations
        self.scenarios = scenarios
        self.selected_shipment_id = selected_shipment_id
        self.selected_checkpoint = selected_checkpoint

    def shipment_details(self, shipment_id: str) -> dict:
        shipment = self.dashboard.data.shipments.get(shipment_id)
        if shipment is None:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        return {"status": "ok", "shipment": {
            "shipment_id": shipment.shipment_id,
            "route_id": shipment.route_id,
            "origin": shipment.origin,
            "destination_port": shipment.destination_port,
            "final_destination": shipment.final_destination,
            "cargo_type": shipment.shipment_type,
            "volume": {"value": float(shipment.volume), "unit": "unspecified in Shipment schema"},
            "scheduled_departure_at_utc": _iso_utc(shipment.scheduled_departure_at),
            "scheduled_port_arrival_at_utc": _iso_utc(shipment.scheduled_port_arrival_at),
            "scheduled_customs_clearance_at_utc": _iso_utc(shipment.scheduled_customs_clearance_at),
            "scheduled_delivery_at_utc": _iso_utc(shipment.scheduled_delivery_at),
            "scenario_id": shipment.scenario_id,
        }, "provenance": "Phase 1 shipment master record; all values are synthetic."}

    def eta_prediction(self, shipment_id: str, checkpoint: str | None = None) -> dict:
        shipment = self.dashboard.data.shipments.get(shipment_id)
        if shipment is None:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        checkpoint = checkpoint or self.selected_checkpoint
        snapshot = self.dashboard.data.snapshots.get((shipment_id, checkpoint))
        if snapshot is None:
            raise ValueError(f"No {checkpoint!r} checkpoint snapshot exists for {shipment_id!r}.")
        prediction = self.dashboard.predictor.predict(snapshot, shipment)
        return {"status": "ok", "prediction": _prediction_dict(prediction),
                "provenance": "Phase 3 adapter using the selected checkpoint snapshot, shipment schedule, route/world configuration, and Phase 3 calibration."}

    def prediction_history(self, shipment_id: str) -> dict:
        if shipment_id not in self.dashboard.data.shipments:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        # This prediction-only frame is deliberately used instead of frame(),
        # which joins actual delivery outcomes for retrospective dashboard use.
        frame = self.dashboard.prediction_frame()
        allowed = ("shipment_id", "checkpoint_type", "checkpoint_at", "predicted_remaining_hours",
                   "predicted_delivery_at", "predicted_total_delay_hours", "probability_delay_over_12h",
                   "remaining_hours_lower", "remaining_hours_upper", "propagation_remaining_hours",
                   "ml_residual_hours", "model_version", "feature_schema_version", "interval_status")
        rows = frame[frame.shipment_id == shipment_id].loc[:, list(allowed)]
        checkpoint_order = ("departure", "port_arrival", "customs_cleared")
        selected_index = checkpoint_order.index(self.selected_checkpoint)
        selected_snapshot = self.dashboard.data.snapshots.get((shipment_id, self.selected_checkpoint))
        if selected_snapshot is None:
            raise ValueError(f"No {self.selected_checkpoint!r} checkpoint snapshot exists for {shipment_id!r}.")
        as_of_checkpoints = list(checkpoint_order[:selected_index + 1])
        later_checkpoints = list(checkpoint_order[selected_index + 1:])
        records = [{"shipment_id": str(row.shipment_id), "checkpoint": str(row.checkpoint_type),
            "prediction_timestamp_utc": _iso_utc(row.checkpoint_at),
            "available_as_of_selected_checkpoint": checkpoint_order.index(str(row.checkpoint_type)) <= selected_index,
            "temporal_label": ("available_as_of_selected_checkpoint" if checkpoint_order.index(str(row.checkpoint_type)) <= selected_index
                               else "retrospective_later_checkpoint"),
            "predicted_remaining_hours": float(row.predicted_remaining_hours),
            "predicted_delivery_at_utc": _iso_utc(row.predicted_delivery_at),
            "predicted_total_delay_hours": float(row.predicted_total_delay_hours),
            "significant_delay_probability": float(row.probability_delay_over_12h),
            "remaining_time_interval_hours": None if pd.isna(row.remaining_hours_lower) else {
                "lower": float(row.remaining_hours_lower), "upper": float(row.remaining_hours_upper)},
            "propagation_remaining_hours": float(row.propagation_remaining_hours),
            "ml_residual_correction_hours": float(row.ml_residual_hours),
            "model_version": str(row.model_version),
            "feature_schema_version": str(row.feature_schema_version),
            "interval_status": str(row.interval_status)} for row in rows.itertuples(index=False)]
        records.sort(key=lambda row: checkpoint_order.index(row["checkpoint"]))
        return {"status": "ok", "shipment_id": shipment_id,
                "selected_checkpoint": self.selected_checkpoint,
                "selected_checkpoint_timestamp_utc": _iso_utc(selected_snapshot.checkpoint_at),
                "temporal_scope": {"as_of_checkpoints": as_of_checkpoints,
                    "later_checkpoints": later_checkpoints,
                    "explanation": "Later checkpoint predictions were unavailable at the selected checkpoint and are retrospective only."},
                "predictions": records,
                "provenance": "Phase 3 adapter reconstructed from prediction snapshots; no actual outcome or target fields are included."}

    def delay_attribution(self, shipment_id: str) -> dict:
        if shipment_id not in self.dashboard.data.shipments:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        checkpoint = self.selected_checkpoint
        driver = self.explanations.prediction_driver_attribution
        shap = self.explanations.shap_local
        if not driver.empty:
            driver = driver[(driver.shipment_id.astype(str) == shipment_id) & (driver.checkpoint.astype(str) == checkpoint)]
        if not shap.empty:
            shap = shap[(shap.shipment_id.astype(str) == shipment_id) & (shap.checkpoint_type.astype(str) == checkpoint)]
        driver_fields = ["shipment_id", "checkpoint", "component", "forecast_used", "forecast_issue_time",
            "forecast_valid_time", "forecast_location", "original_forecast_value", "neutral_reference_value",
            "original_propagation_remaining", "neutralized_propagation_remaining", "effect_hours", "unavailable_reason"]
        shap_fields = ["shipment_id", "checkpoint_type", "feature", "feature_family",
                       "shap_residual_contribution_hours", "direction", "predicted_residual_hours"]
        driver_rows = driver[[field for field in driver_fields if field in driver]].to_dict("records") if not driver.empty else []
        shap_rows = shap[[field for field in shap_fields if field in shap]].to_dict("records") if not shap.empty else []
        return {"status": "ok", "shipment_id": shipment_id, "checkpoint": checkpoint,
            "prediction_driver_attribution": {"status": "available" if driver_rows else "unavailable",
                "reason": None if driver_rows else self.explanations.unavailable.get(
                    "prediction_driver", "No prediction-driver attribution is stored for this shipment/checkpoint."),
                "rows": _jsonable(driver_rows), "units": "hours for effects and propagation durations; raw input value units follow variable"},
            "shap_residual_attribution": {"status": "available" if shap_rows else "unavailable",
                "reason": None if shap_rows else self.explanations.unavailable.get(
                    "shap_local", "No local SHAP record is stored for this shipment/checkpoint."),
                "rows": _jsonable(shap_rows), "units": "hours of residual-model contribution"},
            "retrospective_cause_attribution": {"status": "not_exposed", "reason": "Retrospective simulator cause attribution is ground-truth hindsight and is excluded from prediction-time assistant tools."},
            "provenance": "Phase 3 prediction-driver attribution and local SHAP artifacts, filtered to the selected checkpoint."}

    def query_shipments(self, filters: QueryFilters) -> dict:
        checkpoint = filters.checkpoint or self.selected_checkpoint
        frame = self.dashboard.prediction_frame()
        rows = frame[frame.checkpoint_type == checkpoint]
        if filters.shipment_ids:
            rows = rows[rows.shipment_id.isin(filters.shipment_ids)]
        if filters.predicted_delay_above_hours is not None:
            rows = rows[rows.predicted_total_delay_hours > filters.predicted_delay_above_hours]
        if filters.significant_delay_probability_above is not None:
            rows = rows[rows.probability_delay_over_12h > filters.significant_delay_probability_above]
        if filters.route_id is not None:
            rows = rows[rows.shipment_id.map(lambda sid: self.dashboard.data.shipments[sid].route_id == filters.route_id)]
        if filters.cargo_type is not None:
            rows = rows[rows.shipment_id.map(lambda sid: self.dashboard.data.shipments[sid].shipment_type.casefold() == filters.cargo_type.casefold())]
        rows = rows.sort_values(["probability_delay_over_12h", "predicted_total_delay_hours"], ascending=False).head(filters.limit)
        records = []
        for row in rows.itertuples(index=False):
            shipment = self.dashboard.data.shipments[row.shipment_id]
            records.append({"shipment_id": row.shipment_id, "route_id": shipment.route_id,
                "cargo_type": shipment.shipment_type, "checkpoint": checkpoint,
                "predicted_total_delay_hours": float(row.predicted_total_delay_hours),
                "significant_delay_probability": float(row.probability_delay_over_12h),
                "predicted_delivery_at_utc": _iso_utc(row.predicted_delivery_at),
                "predicted_remaining_hours": float(row.predicted_remaining_hours)})
        return {"status": "ok", "checkpoint": checkpoint, "applied_filters": _jsonable(filters.model_dump()),
                "count": len(records), "results": records,
                "provenance": "Shipment metadata and Phase 3 prediction-only frame; actual outcomes and labels are not queried."}

    def what_if(self, shipment_id: str, checkpoint: str | None, parameters: ScenarioParameters) -> dict:
        shipment = self.dashboard.data.shipments.get(shipment_id)
        if shipment is None:
            raise KeyError(f"Unknown shipment ID {shipment_id!r}.")
        checkpoint = checkpoint or self.selected_checkpoint
        snapshot = self.dashboard.data.snapshots.get((shipment_id, checkpoint))
        if snapshot is None:
            raise ValueError(f"No {checkpoint!r} checkpoint snapshot exists for {shipment_id!r}.")
        result = self.scenarios.run(snapshot, shipment, parameters)
        return {"status": "ok", "label": result.label,
            "original_prediction": _prediction_dict(result.original),
            "modified_prediction": _prediction_dict(result.modified),
            "scenario_parameters": _jsonable(result.parameters),
            "eta_change_hours": float(result.eta_change_hours),
            "remaining_duration_change_hours": float(result.remaining_change_hours),
            "significant_delay_probability_change": float(result.probability_change),
            "stage_changes_hours": _jsonable(result.stage_deltas_hours),
            "source_state_unchanged": bool(result.source_state_unchanged),
            "provenance": "Existing in-memory ScenarioService over cloned checkpoint, route, and world state; no simulator or persistence calls."}


_SHIPMENT_SCHEMA = {"type": "object", "properties": {"shipment_id": {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[A-Za-z0-9_-]+$"}}, "required": ["shipment_id"], "additionalProperties": False}
_CHECKPOINT = {"type": "string", "enum": ["departure", "port_arrival", "customs_cleared"]}
_SCENARIO_PARAMETERS_SCHEMA = {"type": "object", "properties": {
    "port_congestion": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "weather_severity": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "berth_capacity": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
    "customs_workload": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "berth_service_hours": {"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 48},
    "inland_traffic": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
}, "additionalProperties": False}


def _function(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


TOOL_DEFINITIONS = [
    _function("get_shipment_details", "Retrieve validated synthetic shipment manifest and schedule metadata.", _SHIPMENT_SCHEMA),
    _function("get_eta_prediction", "Retrieve a Phase 3 prediction at a selected checkpoint; checkpoint may be omitted to use the selected assistant context.", _object({"shipment_id": _SHIPMENT_SCHEMA["properties"]["shipment_id"], "checkpoint": {"anyOf": [_CHECKPOINT, {"type": "null"}]}}, ("shipment_id",))),
    _function("get_prediction_history", "Retrieve prediction-only history for all three checkpoints without actual outcomes.", _SHIPMENT_SCHEMA),
    _function("get_delay_attribution", "Retrieve separate prediction-driver and local SHAP evidence at the selected assistant checkpoint. Retrospective causes are not exposed.", _SHIPMENT_SCHEMA),
    _function("query_shipments", "Run a bounded query over shipment metadata and predictions only; results are limited to 25.", _object({"filters": _object({
        "checkpoint": {"anyOf": [_CHECKPOINT, {"type": "null"}]},
        "predicted_delay_above_hours": {"type": ["number", "null"], "minimum": -8760, "maximum": 8760},
        "significant_delay_probability_above": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "route_id": {"type": ["string", "null"], "minLength": 1, "maxLength": 80},
        "cargo_type": {"type": ["string", "null"], "minLength": 1, "maxLength": 80},
        "shipment_ids": {"type": ["array", "null"], "minItems": 1, "maxItems": 25, "items": {"type": "string", "minLength": 1, "maxLength": 64}},
        "limit": {"type": "integer", "minimum": 1, "maximum": 25}})}, ("filters",))),
    _function("run_what_if_simulation", "Run the existing in-memory modified-information Phase 3 scenario; this is not an actual outcome.", _object({
        "shipment_id": _SHIPMENT_SCHEMA["properties"]["shipment_id"],
        "checkpoint": {"anyOf": [_CHECKPOINT, {"type": "null"}]},
        "scenario_parameters": _SCENARIO_PARAMETERS_SCHEMA}, ("shipment_id", "scenario_parameters"))),
]


class ToolRegistry:
    """Exactly six statically registered tools with strict argument validation."""

    def __init__(self, service: AgentToolService):
        self.service = service
        self._handlers = {
            "get_shipment_details": (ShipmentArgs, self._shipment),
            "get_eta_prediction": (EtaArgs, self._eta),
            "get_prediction_history": (ShipmentArgs, self._history),
            "get_delay_attribution": (ShipmentArgs, self._attribution),
            "query_shipments": (QueryArgs, self._query),
            "run_what_if_simulation": (WhatIfArgs, self._what_if),
        }

    @property
    def definitions(self) -> list[dict]:
        return json.loads(json.dumps(TOOL_DEFINITIONS))

    def _shipment(self, args): return self.service.shipment_details(args.shipment_id)
    def _eta(self, args): return self.service.eta_prediction(args.shipment_id, args.checkpoint)
    def _history(self, args): return self.service.prediction_history(args.shipment_id)
    def _attribution(self, args): return self.service.delay_attribution(args.shipment_id)
    def _query(self, args): return self.service.query_shipments(args.filters)
    def _what_if(self, args): return self.service.what_if(args.shipment_id, args.checkpoint, args.scenario_parameters)

    def dispatch(self, name: str, arguments: str | dict) -> dict:
        registered = self._handlers.get(name)
        if registered is None:
            return {"status": "error", "code": "unknown_tool", "message": "Tool is not in the approved registry."}
        model, handler = registered
        try:
            if isinstance(arguments, str):
                if len(arguments) > 4096:
                    raise ValueError("Tool argument payload exceeds 4096 characters.")
                decoded = json.loads(arguments)
            else:
                decoded = arguments
            if not isinstance(decoded, dict):
                raise ValueError("Tool arguments must be a JSON object.")
            validated = model.model_validate(decoded)
            result = handler(validated)
            # Verify JSON compatibility at the boundary; fail safely if a new
            # service value cannot be serialized.
            json.dumps(_jsonable(result), allow_nan=False)
            return _jsonable(result)
        except (ValidationError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, ValidationError):
                errors = [{"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
                          for item in exc.errors()[:8]]
                return {"status": "error", "code": "invalid_arguments", "message": "Tool arguments failed validation.", "details": errors}
            return {"status": "error", "code": "tool_error", "message": str(exc)[:300]}
        except Exception:
            return {"status": "error", "code": "tool_error", "message": "Tool execution failed; no result was produced."}
