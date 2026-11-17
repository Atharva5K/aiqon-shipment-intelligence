from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[2]


def test_dashboard_startup_and_navigation_pages_render(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import groq
    provider_initializations = []
    monkeypatch.setattr(groq, "Groq", lambda **kwargs: provider_initializations.append(kwargs))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert "Shipment Network Overview" in [element.value for element in app.title]
    assert len(app.sidebar.radio) == 1
    navigation = app.sidebar.radio[0]
    for page, title in [
        ("Shipment Intelligence", "Shipment Intelligence"),
        ("Explainability", "Explainability"),
        ("What-if Simulation", "What-if Simulation"),
        ("AI Assistant", "AI Assistant"),
        ("Overview", "Shipment Network Overview"),
    ]:
        navigation.set_value(page)
        app = app.run()
        assert not app.exception
        rendered = [element.value for element in app.title] + [element.value for element in app.header]
        assert title in rendered
        if page == "Shipment Intelligence":
            assert len(app.selectbox) >= 2
            assert app.selectbox[0].value == "SHP-0001"
            app.selectbox[0].set_value("SHP-0002")
            app = app.run()
            assert not app.exception
            assert app.selectbox[0].value == "SHP-0002"
        if page == "Explainability":
            assert any("Eligible forecasts" in str(element.value) for element in app.markdown)
            assert any("Configured route/world fallback values" in str(element.value) for element in app.markdown)
            assert app.selectbox[0].label == "Operational-context shipment"
            assert app.selectbox[1].label == "Operational-context checkpoint"
        if page == "AI Assistant":
            assert any("GROQ_API_KEY" in str(element.value) for element in app.info)
            assert len(app.chat_input) == 1
            assert app.chat_input[0].disabled
        if page == "What-if Simulation":
            assert app.slider[0].disabled
            assert app.slider[0].help is None or "complete" in (app.slider[0].help or "").lower()
            app.selectbox[1].set_value("departure")
            app = app.run()
            assert not app.exception
            apply_scenario = next(button for button in app.button if button.label == "Apply scenario")
            apply_scenario.click()
            app = app.run()
            assert not app.exception
            assert any("SIMULATED SCENARIO" in str(element.value) for element in app.markdown)
            assert app.session_state["whatif_context"] == ("SHP-0001", "departure")
            assert app.session_state["whatif_result"].original.shipment_id == "SHP-0001"
            app.selectbox[0].set_value("SHP-0002")
            app = app.run()
            assert not app.exception
            assert "whatif_result" not in app.session_state
            assert "whatif_context" not in app.session_state
            assert any("Adjust supported assumptions" in str(element.value) for element in app.info)
            assert not any("Propagation stage changes" == str(element.value) for element in app.subheader)
            app.selectbox[0].set_value("SHP-0001")
            app = app.run()
            assert not app.exception
            assert "whatif_result" not in app.session_state
            assert "whatif_context" not in app.session_state
            assert any("Adjust supported assumptions" in str(element.value) for element in app.info)
            apply_scenario = next(button for button in app.button if button.label == "Apply scenario")
            apply_scenario.click()
            app = app.run()
            assert not app.exception
            assert app.session_state["whatif_context"] == ("SHP-0001", "departure")
            assert app.session_state["whatif_result"].original.shipment_id == "SHP-0001"
            assert any("SIMULATED SCENARIO" in str(element.value) for element in app.markdown)
            app.selectbox[1].set_value("port_arrival")
            app = app.run()
            assert not app.exception
            assert "whatif_result" not in app.session_state
            assert "whatif_context" not in app.session_state
            traffic_slider = next(slider for slider in app.slider if slider.label == "Inland traffic")
            traffic_slider.set_value(0.99)
            app = app.run()
            next(button for button in app.button if button.label == "Reset scenario controls").click()
            app = app.run()
            assert "whatif_result" not in app.session_state
            assert app.slider[-1].value != 0.99
        navigation = app.sidebar.radio[0]
    assert provider_initializations == []
