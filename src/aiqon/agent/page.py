"""Explicit-submit-only Streamlit interface for the grounded agent."""
from __future__ import annotations

import streamlit as st

from aiqon.agent.config import AgentConfig, AgentConfigurationError, provider_setup_message
from aiqon.agent.loop import AgentLoop
from aiqon.agent.provider import ProviderError, create_provider
from aiqon.agent.tools import AgentToolService, ToolRegistry
from aiqon.dashboard.services.scenario_service import ScenarioService


def _failure_help(error: Exception | str) -> str:
    code = error if isinstance(error, str) else getattr(error, "code", "")
    if code == "authentication":
        return "Groq rejected the key. Verify your key at the Groq Console and update GROQ_API_KEY."
    if code == "permission":
        return "The account cannot access this model. Check account/model availability in the Groq Console."
    if code == "rate_limit":
        return "Groq reported a rate limit or exhausted quota. No automatic retry was attempted; wait for quota reset or review your plan."
    if code == "timeout":
        return "The request timed out. You may submit again when the provider is available."
    if code == "network":
        return "The Groq endpoint was unreachable. Check network access and try again."
    if code == "configuration":
        return str(error)
    if code == "empty_response":
        return "The model returned no text or tool result. Submit a more specific question."
    if code == "tool_round_limit":
        return "The assistant reached its configured tool-call limit. Review the retrieved evidence and narrow the question."
    if code in {"invalid_message", "context_too_large", "malformed_response", "malformed_tool_call", "request_limit", "tool_call_limit"}:
        return "The assistant could not complete the request within its validation or request limits. Try a shorter or narrower question."
    if isinstance(error, AgentConfigurationError):
        return str(error)
    if isinstance(error, ProviderError):
        return str(error)
    return "The assistant could not complete this request. No answer was generated."


def render(dashboard, explanations):
    st.title("AI Assistant")
    st.caption("Optional, tool-grounded investigation assistant. Phase 3 remains the prediction engine; the model does not calculate ETAs.")
    shipment_id = st.selectbox("Shipment", sorted(dashboard.data.shipments), key="ai_shipment")
    checkpoint = st.selectbox("Checkpoint", ["departure", "port_arrival", "customs_cleared"],
                              index=2, key="ai_checkpoint")
    selected_context = (shipment_id, checkpoint)
    if st.session_state.get("ai_context") != selected_context:
        for key in ("ai_chat_history", "ai_last_trace", "ai_last_requests", "ai_last_error", "ai_last_interaction"):
            st.session_state.pop(key, None)
        st.session_state.pop("ai_chat_prompt", None)
        st.session_state["ai_context"] = selected_context

    shipment = dashboard.data.shipments[shipment_id]
    snapshot = dashboard.data.snapshots[(shipment_id, checkpoint)]
    prediction = dashboard.predictor.predict(snapshot, shipment)
    left, right, third = st.columns(3)
    left.metric("Predicted ETA (UTC)", prediction.predicted_delivery_at.strftime("%Y-%m-%d %H:%M"))
    right.metric("Predicted total delay", f"{prediction.predicted_total_delay_hours:+.1f} h")
    third.metric("Significant-delay probability", f"{prediction.probability_delay_over_12h:.1%}")
    st.caption(f"Shipment {shipment_id} · route {shipment.route_id} · {checkpoint} · prediction created {snapshot.checkpoint_at:%Y-%m-%d %H:%M UTC} · {prediction.model_version}")
    st.caption("The probability is an empirical calibration-based approximation. The prediction interval is a calibrated duration range, not a probability distribution.")

    context = {
        "shipment": {"shipment_id": shipment_id, "route_id": shipment.route_id,
                     "origin": shipment.origin, "destination_port": shipment.destination_port,
                     "final_destination": shipment.final_destination,
                     "cargo_type": shipment.shipment_type, "volume": shipment.volume,
                     "volume_unit": "unspecified in Shipment schema",
                     "scheduled_delivery_at_utc": shipment.scheduled_delivery_at.isoformat()},
        "selected_checkpoint": checkpoint,
        "prediction": {
            "checkpoint": checkpoint,
            "prediction_timestamp_utc": prediction.prediction_created_at.isoformat(),
            "predicted_remaining_hours": prediction.predicted_remaining_hours,
            "predicted_delivery_at_utc": prediction.predicted_delivery_at.isoformat(),
            "predicted_total_delay_hours": prediction.predicted_total_delay_hours,
            "significant_delay_probability": prediction.probability_delay_over_12h,
            "probability_status": prediction.probability_status,
            "remaining_time_interval_hours": None if prediction.remaining_hours_lower is None else {
                "lower": prediction.remaining_hours_lower, "upper": prediction.remaining_hours_upper},
            "interval_status": prediction.interval_status,
            "propagation_remaining_hours": prediction.propagation_remaining_hours,
            "ml_residual_correction_hours": prediction.ml_residual_hours,
            "model_version": prediction.model_version,
        },
        "context_provenance": "Shipment schedule metadata plus selected Phase 3 checkpoint snapshot and saved calibration; no future actual outcomes.",
    }

    try:
        config = AgentConfig.from_environment()
        setup_message = provider_setup_message(config)
    except AgentConfigurationError as exc:
        config = None
        setup_message = str(exc)
    if setup_message:
        st.info(setup_message)
        if config and config.provider == "groq" and not config.api_key:
            st.caption("Create a Groq API key in the Groq Console, copy `.env.example` to `.env`, and set GROQ_API_KEY. The evaluator must provide their own key. No key is stored in Streamlit session state.")

    history = list(st.session_state.get("ai_chat_history", []))
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Ask about this shipment or request a bounded analysis…",
                           key="ai_chat_prompt", disabled=bool(setup_message))
    if prompt:
        st.session_state.setdefault("ai_chat_history", []).append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            # Provider construction and inference happen only in this explicit
            # user-submission branch; navigation and rendering make no calls.
            provider = create_provider(config)
            tool_service = AgentToolService(
                dashboard, explanations, ScenarioService(dashboard.predictor),
                selected_shipment_id=shipment_id, selected_checkpoint=checkpoint)
            result = AgentLoop(provider, ToolRegistry(tool_service), config).run(
                prompt, context, history=history)
            st.session_state["ai_last_trace"] = [step.__dict__ for step in result.trace]
            st.session_state["ai_last_requests"] = result.model_requests
            st.session_state["ai_last_error"] = result.error_code
            st.session_state["ai_last_interaction"] = prompt
            if result.text:
                st.session_state["ai_chat_history"].append({"role": "assistant", "content": result.text})
                with st.chat_message("assistant"):
                    st.markdown(result.text)
        except (AgentConfigurationError, ProviderError) as exc:
            st.session_state["ai_last_trace"] = []
            st.session_state["ai_last_requests"] = 0
            st.session_state["ai_last_error"] = getattr(exc, "code", "configuration")
            st.session_state["ai_last_interaction"] = prompt
            st.error(_failure_help(exc))
        # Bound persisted chat to six turns. Tool messages are not put in
        # session state; only user/assistant display text is retained.
        st.session_state["ai_chat_history"] = st.session_state.get("ai_chat_history", [])[-12:]

    if st.session_state.get("ai_last_interaction"):
        error_code = st.session_state.get("ai_last_error")
        if error_code:
            st.error(_failure_help(error_code))
        if st.session_state.get("ai_last_requests") is not None:
            st.caption(f"Model requests for the last interaction: {st.session_state['ai_last_requests']}")
        trace = st.session_state.get("ai_last_trace", [])
        if trace:
            with st.expander("Tool execution trace", expanded=False):
                st.dataframe(trace, use_container_width=True, hide_index=True)
