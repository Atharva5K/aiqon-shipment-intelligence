"""Shipment detail, checkpoint-safe timeline, and prediction history."""
import plotly.graph_objects as go
import streamlit as st

from aiqon.dashboard.charts import apply_light_theme, prediction_history_chart


def _timeline(service, shipment_id, checkpoint, prediction, retrospective):
    all_events = service.data.actual_events[
        service.data.actual_events.shipment_id.astype(str) == shipment_id].copy()
    actual = service.visible_actual_events(shipment_id, checkpoint, retrospective=retrospective)
    figure = go.Figure()
    scheduled = all_events[all_events.scheduled_at.notna()]
    if not scheduled.empty:
        figure.add_trace(go.Scatter(x=scheduled.scheduled_at, y=scheduled.event_type,
            mode="markers", name="Scheduled", marker=dict(color="#64748b", symbol="diamond", size=10),
            hovertemplate="Scheduled %{y}<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>"))
    if not actual.empty:
        figure.add_trace(go.Scatter(x=actual.actual_at, y=actual.event_type,
            mode="markers", name="Actual (visible at checkpoint)" if not retrospective else "Actual (retrospective)",
            marker=dict(color="#059669", symbol="circle", size=11),
            hovertemplate="Actual %{y}<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[prediction.predicted_delivery_at], y=["delivery"],
        mode="markers", name="Predicted ETA", marker=dict(color="#2563eb", symbol="star", size=15),
        hovertemplate="Predicted ETA<br>%{x|%Y-%m-%d %H:%M UTC}<extra></extra>"))
    figure.update_layout(title=("Retrospective completed-journey timeline" if retrospective else
                                f"As-of {checkpoint} timeline — future actual events hidden"),
        xaxis_title="Timestamp (UTC)", yaxis_title="Lifecycle event",
        legend_title="Timestamp type", height=430)
    return apply_light_theme(figure)


def render(service):
    st.title("Shipment Intelligence")
    shipment_ids = sorted(service.data.shipments)
    if not shipment_ids:
        st.info("No shipments are available.")
        return
    shipment_id = st.selectbox("Shipment ID", shipment_ids, key="shipment_id_select")
    shipment = service.data.shipments[shipment_id]
    checkpoint = st.selectbox("Prediction checkpoint",
        ["departure", "port_arrival", "customs_cleared"], index=2, key="shipment_checkpoint_select")
    snapshot = service.data.snapshots[(shipment_id, checkpoint)]
    history = service.shipment_history(shipment_id)
    current = history[history.checkpoint_type == checkpoint].iloc[0]
    prediction = service.predictor.predict(snapshot, shipment)

    st.subheader("Shipment manifest")
    cols = st.columns(4)
    cols[0].write(f"**Route:** {shipment.route_id}")
    cols[1].write(f"**Origin:** {shipment.origin}")
    cols[2].write(f"**Destination port:** {shipment.destination_port}")
    cols[3].write(f"**Final destination:** {shipment.final_destination}")
    cols = st.columns(4)
    cols[0].write(f"**Cargo type:** {shipment.shipment_type}")
    cols[1].write(f"**Volume:** {shipment.volume:.2f}")
    cols[2].write(f"**Scenario ID:** {shipment.scenario_id}")
    cols[3].write(f"**Scheduled departure (UTC):** {shipment.scheduled_departure_at:%Y-%m-%d %H:%M}")
    st.write(f"**Scheduled delivery (UTC):** {shipment.scheduled_delivery_at:%Y-%m-%d %H:%M}")

    st.subheader("Current Phase 3 prediction")
    first, second, third, fourth = st.columns(4)
    first.metric("Predicted remaining", f"{prediction.predicted_remaining_hours:.1f} h")
    second.metric("Predicted final ETA", prediction.predicted_delivery_at.strftime("%Y-%m-%d %H:%M UTC"))
    third.metric("Predicted total delay", f"{prediction.predicted_total_delay_hours:+.1f} h")
    fourth.metric("Significant-delay probability", f"{prediction.probability_delay_over_12h:.1%}")
    if prediction.remaining_hours_lower is None:
        st.warning(f"90% prediction interval unavailable: {prediction.interval_status}.")
    else:
        st.metric("90% remaining-time interval",
                  f"{prediction.remaining_hours_lower:.1f}–{prediction.remaining_hours_upper:.1f} h")
    left, right, third_col = st.columns(3)
    left.metric("Propagation baseline", f"{prediction.propagation_remaining_hours:.1f} h")
    right.metric("ML residual correction", f"{prediction.ml_residual_hours:+.1f} h")
    third_col.caption(f"Model {prediction.model_version} · schema {prediction.feature_schema_version}")
    st.caption("Probability is an empirical calibration-based approximation, not a guaranteed calibrated probability. The conformal interval is not a probability distribution.")

    st.subheader("Event timeline")
    retrospective = st.checkbox("Show completed journey retrospectively (includes future actual events)", value=False,
                                key="retrospective_timeline")
    st.plotly_chart(_timeline(service, shipment_id, checkpoint, prediction, retrospective),
                    use_container_width=True)
    if not retrospective:
        st.info("As-of mode shows scheduled events and only actual events allowed by this checkpoint's visibility contract.")
    else:
        st.warning("Retrospective view includes future actuals and must not be used as a prediction-time view.")

    st.subheader("Prediction history across checkpoints")
    st.plotly_chart(prediction_history_chart(history), use_container_width=True)
    display = history[["checkpoint_type", "checkpoint_at", "predicted_delivery_at",
        "predicted_remaining_hours", "predicted_total_delay_hours",
        "probability_delay_over_12h", "remaining_hours_lower", "remaining_hours_upper"]].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(f"The highlighted prediction was created at {current.checkpoint_at:%Y-%m-%d %H:%M UTC}. History rows are independently reconstructed from each checkpoint snapshot.")

