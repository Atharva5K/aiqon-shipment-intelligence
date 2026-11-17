"""Interactive, read-only modified-information prediction scenarios."""
import pandas as pd
import streamlit as st

from aiqon.prediction.features import build_features
from aiqon.dashboard.services.scenario_service import ScenarioParameters


def _interval(prediction):
    if prediction.remaining_hours_lower is None:
        return "Unavailable"
    return f"{prediction.remaining_hours_lower:.1f}–{prediction.remaining_hours_upper:.1f} h"


def _first_present(*values):
    """Return the first available value, preserving meaningful zeros."""
    return next((value for value in values if value is not None), None)


def render(service, scenario_service):
    st.title("What-if Simulation")
    st.warning("SIMULATED SCENARIO — NOT AN ACTUAL SHIPMENT OUTCOME", icon="⚠️")
    st.caption("This changes only the selected checkpoint's information/configuration in memory. It does not run the ground-truth simulator or write prediction history.")
    shipment_id = st.selectbox("Shipment ID", sorted(service.data.shipments), key="whatif_shipment")
    checkpoint = st.selectbox("Prediction checkpoint",
        ["departure", "port_arrival", "customs_cleared"], index=2, key="whatif_checkpoint")
    current_selection = (shipment_id, checkpoint)
    previous_selection = st.session_state.get("whatif_selection")
    if previous_selection != current_selection:
        st.session_state.pop("whatif_result", None)
        st.session_state.pop("whatif_context", None)
        st.session_state["whatif_selection"] = current_selection
    shipment = service.data.shipments[shipment_id]
    snapshot = service.data.snapshots[(shipment_id, checkpoint)]
    route = scenario_service.predictor.routes[shipment.route_id]
    world = scenario_service.predictor.bundle["world"]
    features = build_features(snapshot, shipment, route, world)
    support = scenario_service.parameter_support(snapshot, shipment, route, world)
    defaults = {
        "port_congestion": _first_present(features.get("forecast_port_congestion"), features.get("observed_port_congestion"), route.congestion.baseline),
        "weather_severity": _first_present(features.get("forecast_port_weather"), features.get("observed_port_weather"), 0.5),
        "berth_capacity": route.berth_capacity,
        "customs_workload": _first_present(features.get("forecast_customs_workload"), features.get("observed_customs_workload"), route.customs.workload_baseline),
        "berth_service_hours": world.berth_service_base_hours,
        "inland_traffic": _first_present(features.get("forecast_inland_traffic"), features.get("observed_inland_traffic"), route.inland.traffic_baseline),
    }
    keys = {name: f"whatif_{name}" for name in defaults}
    if st.button("Reset scenario controls", key="whatif_reset"):
        st.session_state.pop("whatif_result", None)
        st.session_state.pop("whatif_context", None)
        for key in keys.values():
            st.session_state.pop(key, None)
        st.rerun()

    st.subheader("Current operational context")
    st.write(f"**Route:** {route.route_id} · **Checkpoint:** {checkpoint} · **As-of time:** {snapshot.checkpoint_at:%Y-%m-%d %H:%M UTC}")
    st.caption("Disabled controls are unavailable at this checkpoint. Port weather is a checkpoint-visible ML feature only; propagation does not consume it. The current saved residual model has zero split importance for port-weather inputs, so changing weather does not shift its residual estimate.")
    with st.form("scenario_parameters"):
        left, right = st.columns(2)
        with left:
            congestion = st.slider("Port congestion", 0.0, 1.0, float(defaults["port_congestion"]),
                                   0.01, key=keys["port_congestion"], disabled=not support["port_congestion"][0],
                                   help=support["port_congestion"][1])
            weather = st.slider("Port weather severity", 0.0, 1.0, float(defaults["weather_severity"]),
                                0.01, key=keys["weather_severity"], disabled=not support["weather_severity"][0],
                                help=support["weather_severity"][1])
            berth_capacity = st.slider("Berth capacity", 1, 10, int(route.berth_capacity), 1,
                key=keys["berth_capacity"], disabled=not support["berth_capacity"][0],
                help=support["berth_capacity"][1])
        with right:
            workload = st.slider("Customs workload", 0.0, 1.0, float(defaults["customs_workload"]),
                0.01, key=keys["customs_workload"], disabled=not support["customs_workload"][0],
                help=support["customs_workload"][1])
            service_hours = st.slider("Mean berth service-time assumption (hours)", 1.0, 24.0,
                float(min(24.0, max(1.0, defaults["berth_service_hours"]))), 0.25,
                key=keys["berth_service_hours"], disabled=not support["berth_service_hours"][0],
                help=support["berth_service_hours"][1])
            traffic = st.slider("Inland traffic", 0.0, 1.0, float(defaults["inland_traffic"]),
                0.01, key=keys["inland_traffic"], disabled=not support["inland_traffic"][0],
                help=support["inland_traffic"][1])
        apply = st.form_submit_button("Apply scenario")

    if apply:
        values = {
            "port_congestion": congestion if support["port_congestion"][0] else None,
            "weather_severity": weather if support["weather_severity"][0] else None,
            "berth_capacity": berth_capacity if support["berth_capacity"][0] else None,
            "customs_workload": workload if support["customs_workload"][0] else None,
            "berth_service_hours": service_hours if support["berth_service_hours"][0] else None,
            "inland_traffic": traffic if support["inland_traffic"][0] else None,
        }
        try:
            result = scenario_service.run(snapshot, shipment, ScenarioParameters(**values))
            st.session_state["whatif_result"] = result
            st.session_state["whatif_context"] = (shipment_id, checkpoint)
        except Exception as exc:
            st.error(f"Scenario could not be calculated: {exc}")

    result = st.session_state.get("whatif_result")
    context = st.session_state.get("whatif_context")
    if result is None or context != (shipment_id, checkpoint):
        st.info("Adjust supported assumptions and apply a scenario to compare Phase 3 predictions.")
        for name, (enabled, message) in support.items():
            if not enabled:
                st.caption(f"{name}: unavailable — {message}")
        return

    st.markdown(f"### {result.label}")
    original, modified = result.original, result.modified
    comparison = pd.DataFrame([
        {"Measure": "Predicted final ETA (UTC)", "Original": original.predicted_delivery_at.strftime("%Y-%m-%d %H:%M UTC"),
         "Modified": modified.predicted_delivery_at.strftime("%Y-%m-%d %H:%M UTC")},
        {"Measure": "Predicted remaining duration", "Original": f"{original.predicted_remaining_hours:.2f} h",
         "Modified": f"{modified.predicted_remaining_hours:.2f} h"},
        {"Measure": "Predicted total delay", "Original": f"{original.predicted_total_delay_hours:+.2f} h",
         "Modified": f"{modified.predicted_total_delay_hours:+.2f} h"},
        {"Measure": "Significant-delay probability", "Original": f"{original.probability_delay_over_12h:.1%}",
         "Modified": f"{modified.probability_delay_over_12h:.1%}"},
        {"Measure": "90% remaining-time interval", "Original": _interval(original),
         "Modified": _interval(modified)},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    a.metric("ETA change", f"{result.eta_change_hours:+.2f} h")
    b.metric("Remaining duration change", f"{result.remaining_change_hours:+.2f} h")
    c.metric("Significant-delay probability change", f"{result.probability_change:+.1%}")
    stage = pd.DataFrame([{"stage": key,
        "original_hours": original.propagation_components[key],
        "modified_hours": modified.propagation_components[key],
        "change_hours": result.stage_deltas_hours[key]}
        for key in result.stage_deltas_hours])
    st.subheader("Propagation stage changes")
    st.dataframe(stage, use_container_width=True, hide_index=True)
    st.caption("Predictions use only the selected checkpoint snapshot, shipment schedule metadata, route/world assumptions, and eligible checkpoint forecasts. No future actual event or condition is used.")
    st.caption("Scenario changes are temporary session state only; they are not appended to historical prediction records.")

