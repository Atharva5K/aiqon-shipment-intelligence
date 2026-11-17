"""Three deliberately distinct Phase 3 explanation views."""
import streamlit as st

from aiqon.dashboard.charts import shap_global_chart, shap_local_chart
from aiqon.prediction.features import build_features
from aiqon.prediction.propagation import estimate_remaining_duration


def _empty(unavailable, category, message):
    if category in unavailable:
        st.info(unavailable[category])
    else:
        st.info(message)


def _operational_context(service, shipment_id, checkpoint):
    shipment = service.data.shipments[shipment_id]
    snapshot = service.data.snapshots[(shipment_id, checkpoint)]
    route = service.predictor.routes[shipment.route_id]
    world = service.predictor.bundle["world"]
    features = build_features(snapshot, shipment, route, world)
    st.subheader("Selected checkpoint context")
    st.write(f"**Shipment:** {shipment_id} · **Route:** {route.route_id} · **Checkpoint:** {checkpoint} · **Checkpoint time (UTC):** {snapshot.checkpoint_at:%Y-%m-%d %H:%M:%S %Z}")

    env = snapshot.current_environment
    stage = snapshot.observed_stage_fields
    observed = [
        {"Operational input": "Port congestion", "Observed as of checkpoint": env.get("port_congestion"), "Source": "Checkpoint observation" if env.get("port_congestion") is not None else "Not observed"},
        {"Operational input": "Port weather", "Observed as of checkpoint": env.get("port_weather"), "Source": "Checkpoint observation" if env.get("port_weather") is not None else "Not observed"},
        {"Operational input": "Origin weather", "Observed as of checkpoint": env.get("origin_weather"), "Source": "Checkpoint observation" if env.get("origin_weather") is not None else "Not observed"},
        {"Operational input": "Queue length", "Observed as of checkpoint": stage.get("observed_queue_length"), "Source": "Checkpoint stage observation" if stage.get("observed_queue_length") is not None else "Not observed"},
        {"Operational input": "Queue state", "Observed as of checkpoint": env.get("queue_state"), "Source": "Checkpoint observation" if env.get("queue_state") is not None else "Not observed"},
        {"Operational input": "Berth capacity available", "Observed as of checkpoint": stage.get("berth_capacity_available"), "Source": "Checkpoint stage observation" if stage.get("berth_capacity_available") is not None else "Not observed"},
        {"Operational input": "Customs workload", "Observed as of checkpoint": env.get("customs_workload"), "Source": "Checkpoint observation" if env.get("customs_workload") is not None else "Not observed"},
        {"Operational input": "Inland traffic", "Observed as of checkpoint": env.get("inland_traffic"), "Source": "Checkpoint observation" if env.get("inland_traffic") is not None else "Not observed"},
    ]
    st.dataframe(observed, use_container_width=True, hide_index=True)

    raw_trace = {}
    estimate_remaining_duration(snapshot, route, world, forecast_trace=raw_trace)
    st.markdown("**Eligible forecasts**")
    forecast_rows = []
    for variable in ("transit_congestion", "port_congestion", "port_weather", "customs_workload", "inland_traffic"):
        selected_value = features.get(f"forecast_{variable}")
        if selected_value is None:
            continue
        candidates = [forecast for forecast in snapshot.forecasts
                      if forecast.forecast_variable == variable
                      and forecast.forecast_issued_at <= snapshot.checkpoint_at
                      and forecast.forecast_value == selected_value]
        if not candidates:
            continue
        forecast = max(candidates, key=lambda item: (item.forecast_issued_at, item.forecast_valid_at))
        forecast_rows.append({"Variable": variable, "Value": forecast.forecast_value,
            "Issued at (UTC)": forecast.forecast_issued_at,
            "Valid at (UTC)": forecast.forecast_valid_at,
            "Consumed by propagation": variable in raw_trace and raw_trace[variable]["source"] == "forecast"})
    if forecast_rows:
        st.dataframe(forecast_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No eligible forecast feature is selected for this checkpoint prediction.")

    st.markdown("**Configured route/world fallback values**")
    fallback_rows = [
        {"Fallback": "Port congestion baseline", "Value": route.congestion.baseline,
         "Use": raw_trace.get("port_congestion", {}).get("source", "Not requested by propagation")},
        {"Fallback": "Customs workload baseline", "Value": route.customs.workload_baseline,
         "Use": raw_trace.get("customs_workload", {}).get("source", "Not requested by propagation")},
        {"Fallback": "Inland traffic baseline", "Value": route.inland.traffic_baseline,
         "Use": raw_trace.get("inland_traffic", {}).get("source", "Not requested by propagation")},
        {"Fallback": "Configured berth capacity", "Value": route.berth_capacity, "Use": "Route configuration"},
        {"Fallback": "Berth service base duration (hours)", "Value": world.berth_service_base_hours, "Use": "World configuration"},
        {"Fallback": "Configured inland travel duration (hours)", "Value": route.inland.travel_hours, "Use": "Route configuration"},
    ]
    st.dataframe(fallback_rows, use_container_width=True, hide_index=True)
    st.caption("Observations and forecasts above come only from this validated checkpoint snapshot. Configured values are labeled as fallbacks/configuration; they are not presented as observed conditions. Future actual conditions are not shown.")


def render(service, explanations):
    st.title("Explainability")
    st.caption("These explanation types answer different questions and are not interchangeable.")
    context_col1, context_col2 = st.columns(2)
    with context_col1:
        shipment_id = st.selectbox("Operational-context shipment", sorted(service.data.shipments), key="explain_context_shipment")
    with context_col2:
        checkpoint = st.selectbox("Operational-context checkpoint", ["departure", "port_arrival", "customs_cleared"],
                                  key="explain_context_checkpoint")
    _operational_context(service, shipment_id, checkpoint)

    st.divider()
    st.header("Retrospective Cause Attribution")
    st.caption("Ground-truth simulator reruns explain a completed synthetic outcome. They do not describe what the predictor knew at a checkpoint and do not establish real-world causality.")
    retrospective = explanations.retrospective_cause_attribution
    if retrospective.empty:
        _empty(explanations.unavailable, "retrospective", "No retrospective attribution rows are available.")
    else:
        shipment_ids = ["All", *sorted(retrospective.shipment_id.astype(str).unique())]
        selected = st.selectbox("Retrospective shipment", shipment_ids, key="explain_retrospective_shipment")
        shown = retrospective if selected == "All" else retrospective[retrospective.shipment_id.astype(str) == selected]
        columns = [c for c in ("shipment_id", "mechanism", "original_actual_delivery_at",
            "counterfactual_actual_delivery_at", "retrospective_cause_effect_hours") if c in shown]
        st.dataframe(shown[columns].copy(), use_container_width=True, hide_index=True)
        st.caption("Effect is original actual delivery minus the matched-seed simulator counterfactual. One-at-a-time effects may interact.")

    st.header("Prediction Driver Attribution")
    st.caption("As-of propagation sensitivity neutralizes only a forecast vintage actually consumed by the propagation estimator. It is not ground-truth causal attribution.")
    driver = explanations.prediction_driver_attribution
    if driver.empty:
        _empty(explanations.unavailable, "prediction_driver", "No prediction-driver attribution rows are available.")
    else:
        components = ["All", *sorted(driver.component.astype(str).unique())]
        component = st.selectbox("Forecast component", components, key="explain_driver_component")
        shown = driver if component == "All" else driver[driver.component == component]
        fields = [c for c in ("shipment_id", "checkpoint", "component", "forecast_used",
            "forecast_issue_time", "forecast_valid_time", "forecast_location",
            "original_forecast_value", "neutral_reference_value",
            "original_propagation_remaining", "neutralized_propagation_remaining",
            "effect_hours", "unavailable_reason") if c in shown]
        st.dataframe(shown[fields].copy(), use_container_width=True, hide_index=True)
        used = int(shown.forecast_used.fillna(False).sum()) if "forecast_used" in shown else 0
        st.metric("Rows where an eligible forecast was actually consumed", used)
        st.caption("Unavailable or unsupported components are reported explicitly; no effect is inferred for a forecast the propagation model did not use.")

    st.header("ML Residual Feature Contribution (SHAP)")
    st.caption("SHAP explains the Gradient Boosting residual correction, not the complete propagation estimate or a simulator cause.")
    if explanations.shap_global.empty:
        _empty(explanations.unavailable, "shap_global", "Global SHAP records are unavailable.")
    else:
        st.plotly_chart(shap_global_chart(explanations.shap_global), use_container_width=True)
        st.dataframe(explanations.shap_global.head(20), use_container_width=True, hide_index=True)
    if explanations.shap_figure:
        st.image(str(explanations.shap_figure), caption="Phase 3 saved global residual SHAP summary")
    if explanations.shap_local.empty:
        _empty(explanations.unavailable, "shap_local", "No stored local SHAP records are available.")
    else:
        shipment = st.selectbox("Local SHAP shipment",
            sorted(explanations.shap_local.shipment_id.astype(str).unique()), key="explain_local_shipment")
        local_ship = explanations.shap_local[explanations.shap_local.shipment_id.astype(str) == shipment]
        checkpoints = sorted(local_ship.checkpoint_type.astype(str).unique())
        checkpoint = st.selectbox("Local SHAP checkpoint", checkpoints, key="explain_local_checkpoint")
        local = local_ship[local_ship.checkpoint_type == checkpoint]
        st.plotly_chart(shap_local_chart(local), use_container_width=True)
        st.dataframe(local.sort_values("shap_residual_contribution_hours", ascending=False),
                     use_container_width=True, hide_index=True)

