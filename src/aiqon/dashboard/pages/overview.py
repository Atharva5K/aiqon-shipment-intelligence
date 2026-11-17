"""Overview page for synthetic shipment outcomes and Phase 3 predictions."""
import streamlit as st

from aiqon.dashboard.charts import (
    actual_delay_distribution, actual_predicted_delay, checkpoint_performance_chart,
    predicted_risk_distribution, route_summary, scheduled_predicted_eta,
)
from aiqon.dashboard.metrics import filter_dashboard_frame, overview_kpis, shipment_level_frame


def render(service):
    st.title("Shipment Network Overview")
    st.caption("Historical outcomes and Phase 3 predictions are shown separately. Shipment-level KPIs use one selected checkpoint.")
    full = service.frame()
    if full.empty:
        st.info("No prediction rows are available for the selected data root.")
        return

    with st.sidebar:
        st.header("Overview filters")
        route = st.selectbox("Route", ["All", *sorted(full.route_id.unique())], key="overview_route")
        scenario = st.selectbox("Scenario", ["All", *sorted(full.scenario_id.unique())], key="overview_scenario")
        checkpoint = st.selectbox("Prediction checkpoint", ["departure", "port_arrival", "customs_cleared"],
                                  index=2, key="overview_checkpoint")
        delay_status = st.selectbox("Historical actual delay status",
            ["All historical statuses", "On time / early", "Delayed (> 0h)", "Significant delay (> 12h)"],
            key="overview_delay_status")
    selected = filter_dashboard_frame(full, route=route, scenario=scenario,
                                      delay_status=delay_status, checkpoint=checkpoint)
    shipments = shipment_level_frame(selected, checkpoint)
    if shipments.empty:
        st.info("No shipments match these filters. Adjust the route, scenario, or historical delay status.")
        return
    kpi = overview_kpis(shipments)
    st.caption(f"KPIs below count unique shipments with a {checkpoint} prediction; {len(shipments)} rows = {shipments.shipment_id.nunique()} shipments.")
    cards = st.columns(5)
    cards[0].metric("Unique shipments", f"{kpi['unique_shipments']:,}")
    cards[1].metric("Historically delayed", f"{kpi['actual_delayed_shipments']:,}")
    cards[2].metric("Historical >12h delay", f"{kpi['actual_significant_delay_shipments']:,}")
    cards[3].metric("Mean predicted final delay", f"{kpi['predicted_mean_delay_hours']:.1f} h")
    cards[4].metric("Median predicted final delay", f"{kpi['predicted_median_delay_hours']:.1f} h")
    st.caption("Historical counts use completed simulator outcomes. Predicted delay is computed from the selected checkpoint's predicted ETA minus scheduled delivery.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(actual_delay_distribution(shipments), use_container_width=True)
    with right:
        st.plotly_chart(predicted_risk_distribution(shipments), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(scheduled_predicted_eta(shipments), use_container_width=True)
    with right:
        st.plotly_chart(route_summary(shipments), use_container_width=True)

    st.subheader("Checkpoint-level prediction performance")
    # This comparison intentionally spans checkpoints; route, scenario and
    # retrospective delay filters are applied consistently, while this chart
    # compares all three checkpoint types by design.
    checkpoint_rows = filter_dashboard_frame(full, route=route, scenario=scenario,
                                              delay_status=delay_status, checkpoint="All checkpoints")
    st.plotly_chart(checkpoint_performance_chart(checkpoint_rows), use_container_width=True)
    st.caption("MAE is calculated on checkpoint snapshots; each shipment contributes up to three rows. This chart is a retrospective diagnostic.")
    st.plotly_chart(actual_predicted_delay(shipments), use_container_width=True)
    st.caption("Actual versus predicted final delay uses one prediction per shipment at the selected checkpoint.")
    st.caption("All shipment records, route profiles, operational conditions and outcomes are synthetically generated for this project. Results do not represent actual logistics operations.")

