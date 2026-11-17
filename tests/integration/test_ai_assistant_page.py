from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from aiqon.agent.provider import NormalizedResponse


ROOT = Path(__file__).resolve().parents[2]


class StubProvider:
    def __init__(self):
        self.requests = 0

    def complete(self, **kwargs):
        self.requests += 1
        return NormalizedResponse("Grounded test response.")


def test_explicit_submission_is_the_only_inference_trigger_and_context_change_clears_chat(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-dummy")
    from aiqon.agent import page as assistant_page

    provider = StubProvider()
    factory_calls = []

    def fake_factory(config):
        factory_calls.append(config.model)
        return provider

    monkeypatch.setattr(assistant_page, "create_provider", fake_factory)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert provider.requests == 0
    assert factory_calls == []
    app.sidebar.radio[0].set_value("AI Assistant")
    app = app.run()
    assert not app.exception
    assert provider.requests == 0
    assert factory_calls == []

    app.chat_input[0].set_value("Explain this shipment prediction.")
    app = app.run()
    assert not app.exception
    assert provider.requests == 1
    assert factory_calls == ["openai/gpt-oss-20b"]
    assert any(turn["content"] == "Grounded test response." for turn in app.session_state["ai_chat_history"])

    app = app.run()
    assert not app.exception
    assert provider.requests == 1

    app.selectbox[0].set_value("SHP-0002")
    app = app.run()
    assert not app.exception
    assert provider.requests == 1
    assert "ai_chat_history" not in app.session_state
    assert app.session_state["ai_context"][0] == "SHP-0002"
