"""Plotly chart factories with explicit shipment/snapshot units."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from aiqon.dashboard.metrics import checkpoint_performance


def apply_light_theme(fig):
    """Apply a consistent, readable light palette without changing chart data."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#F7F9FC",
        plot_bgcolor="#FFFFFF",
        font={"color": "#172033", "family": "Arial, sans-serif"},
        title_font={"color": "#172033"},
        legend={"font": {"color": "#526078"}},
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#E2E8F0",
            "font": {"color": "#172033"},
        },
    )
    fig.update_xaxes(
        title_font={"color": "#526078"},
        tickfont={"color": "#526078"},
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
    )
    fig.update_yaxes(
        title_font={"color": "#526078"},
        tickfont={"color": "#526078"},
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
    )
    return fig


def actual_delay_distribution(frame):
    fig = px.histogram(frame.drop_duplicates("shipment_id"), x="actual_delay_hours", nbins=28,
                       title="Historical actual delivery delay (unique shipments)")
    fig.update_layout(xaxis_title="Actual delay from scheduled delivery (hours)", yaxis_title="Shipments")
    return apply_light_theme(fig)


def predicted_risk_distribution(frame):
    fig = px.histogram(frame.drop_duplicates("shipment_id"), x="probability_delay_over_12h",
                       nbins=20, range_x=[0, 1], title="Empirical calibration-based significant-delay risk")
    fig.update_layout(xaxis_title="Probability approximation", yaxis_title="Shipments")
    return apply_light_theme(fig)


def scheduled_predicted_eta(frame):
    fig = px.scatter(frame.drop_duplicates("shipment_id"), x="scheduled_delivery_at",
                     y="predicted_delivery_at", color="route_id", hover_name="shipment_id",
                     title="Scheduled versus predicted ETA (one selected checkpoint per shipment)")
    fig.update_layout(xaxis_title="Scheduled delivery (UTC)", yaxis_title="Predicted delivery (UTC)")
    return apply_light_theme(fig)


def route_summary(frame):
    if frame.empty:
        return apply_light_theme(go.Figure())
    summary = (frame.drop_duplicates("shipment_id").groupby("route_id", as_index=False)
               .agg(shipments=("shipment_id", "nunique"),
                    mean_actual_delay_hours=("actual_delay_hours", "mean"),
                    mean_predicted_delay_hours=("predicted_total_delay_hours", "mean")))
    fig = go.Figure()
    fig.add_bar(x=summary.route_id, y=summary.mean_actual_delay_hours, name="Historical actual mean delay")
    fig.add_bar(x=summary.route_id, y=summary.mean_predicted_delay_hours, name="Predicted mean delay")
    fig.update_layout(barmode="group", title="Route summary (unique shipments)",
                      xaxis_title="Synthetic route", yaxis_title="Mean delay (hours)")
    return apply_light_theme(fig)


def checkpoint_performance_chart(frame):
    summary = checkpoint_performance(frame)
    fig = px.bar(summary, x="checkpoint_type", y="mae_hours", color="checkpoint_type",
                 hover_data=["sample_count"], title="Checkpoint prediction error (shipment snapshots)")
    fig.update_layout(showlegend=False, xaxis_title="Prediction checkpoint", yaxis_title="MAE (hours)")
    return apply_light_theme(fig)


def actual_predicted_delay(frame):
    fig = px.scatter(frame, x="actual_delay_hours", y="predicted_total_delay_hours",
                     color="checkpoint_type", hover_name="shipment_id",
                     title="Historical actual versus predicted final delay (checkpoint snapshots)")
    bounds = []
    if not frame.empty:
        bounds = [min(frame.actual_delay_hours.min(), frame.predicted_total_delay_hours.min()),
                  max(frame.actual_delay_hours.max(), frame.predicted_total_delay_hours.max())]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", name="Equal delay"))
    fig.update_layout(xaxis_title="Historical actual final delay (hours)",
                      yaxis_title="Predicted final delay (hours)")
    return apply_light_theme(fig)


def prediction_history_chart(frame):
    fig = go.Figure()
    ordered = frame.sort_values("checkpoint_at")
    fig.add_trace(go.Scatter(x=ordered.checkpoint_at, y=ordered.predicted_delivery_at,
        mode="lines+markers", name="Predicted ETA", line=dict(color="#2563eb", width=3)))
    if not ordered.empty:
        fig.add_hline(y=ordered.scheduled_delivery_at.iloc[0], line_dash="dash",
                      line_color="#6b7280", annotation_text="Scheduled delivery")
    fig.update_layout(title="Prediction history as new checkpoints become available",
        xaxis_title="Prediction checkpoint time (UTC)", yaxis_title="Predicted delivery (UTC)")
    return apply_light_theme(fig)


def shap_global_chart(frame, limit=20):
    if frame.empty:
        return apply_light_theme(go.Figure())
    selected = frame.sort_values("mean_absolute_shap", ascending=True).tail(limit)
    fig = go.Figure(go.Bar(x=selected.mean_absolute_shap, y=selected.feature,
        orientation="h", marker_color="#6d5ce8", customdata=selected[["mean_positive_shap", "mean_negative_shap"]],
        hovertemplate="%{y}<br>Mean |SHAP|: %{x:.2f}h<br>Mean positive: %{customdata[0]:.2f}h<br>Mean negative: %{customdata[1]:.2f}h<extra></extra>"))
    fig.update_layout(title="Global ML residual feature contribution",
                      xaxis_title="Mean absolute SHAP contribution (hours)", yaxis_title="Feature")
    return apply_light_theme(fig)


def shap_local_chart(frame):
    if frame.empty:
        return apply_light_theme(go.Figure())
    ordered = frame.sort_values("shap_residual_contribution_hours")
    colors = ["#dc2626" if value < 0 else "#059669" for value in ordered.shap_residual_contribution_hours]
    fig = go.Figure(go.Bar(x=ordered.shap_residual_contribution_hours, y=ordered.feature,
                           orientation="h", marker_color=colors))
    fig.update_layout(title="Local ML residual contribution (selected shipment/checkpoint)",
                      xaxis_title="SHAP contribution (hours)", yaxis_title="Feature")
    return apply_light_theme(fig)

