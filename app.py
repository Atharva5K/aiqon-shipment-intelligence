"""Phase 4 Streamlit dashboard entry point."""
from __future__ import annotations

import streamlit as st

from aiqon.agent import page as ai_assistant
from aiqon.dashboard.config import DashboardPaths
from aiqon.dashboard.data_loader import DashboardDataError, load_dashboard_data
from aiqon.dashboard.pages import explainability, overview, shipment_intelligence, what_if
from aiqon.dashboard.services.dashboard_service import DashboardService
from aiqon.dashboard.services.explanation_service import load_explanations
from aiqon.dashboard.services.prediction_adapter import DashboardArtifactError, Phase3PredictionAdapter
from aiqon.dashboard.services.scenario_service import ScenarioService


st.set_page_config(page_title="Aiqon Shipment Intelligence", page_icon="🚢", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.7rem; max-width: 1500px; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; padding: 14px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

paths = DashboardPaths()
st.sidebar.title("Aiqon · Shipment Intelligence")
st.sidebar.warning("Synthetic research data. Not live operational information.")
page = st.sidebar.radio("Navigate", ["Overview", "Shipment Intelligence", "Explainability", "What-if Simulation", "AI Assistant"],
                        key="dashboard_page")
st.sidebar.caption("Phase 3 hybrid predictions · read-only dashboard")


@st.cache_data(show_spinner="Loading synthetic shipments and checkpoint snapshots…")
def cached_data(root: str):
    return load_dashboard_data(root)


@st.cache_resource(show_spinner="Checking Phase 3 model and calibration artifacts…")
def cached_predictor(bundle: str, calibration: str, metadata: str):
    return Phase3PredictionAdapter.load(bundle, calibration, metadata)


try:
    with st.spinner("Validating Phase 3 dataset and artifacts…"):
        data = cached_data(str(paths.root))
        predictor = cached_predictor(str(paths.artifacts / "phase3_bundle.joblib"),
                                     str(paths.artifacts / "phase3_calibration.joblib"),
                                     str(paths.artifacts / "phase3_metadata.json"))
        service = DashboardService(data, predictor)
        explanations = load_explanations(paths.metrics, paths.figures)
except (DashboardDataError, DashboardArtifactError, OSError, ValueError) as exc:
    st.error(f"Dashboard could not start: {exc}")
    st.info("Generate the Phase 1 dataset and Phase 3 artifacts, then restart the dashboard.")
    st.stop()

if page == "Overview":
    overview.render(service)
elif page == "Shipment Intelligence":
    shipment_intelligence.render(service)
elif page == "Explainability":
    explainability.render(service, explanations)
elif page == "What-if Simulation":
    what_if.render(service, ScenarioService(predictor))
else:
    ai_assistant.render(service, explanations)
