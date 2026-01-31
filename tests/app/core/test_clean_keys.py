import pytest
import streamlit as st


@pytest.mark.integration
def test_integration_create_keys() -> None:
    """Integration test for creating session state keys."""
    state = st.session_state

    state["samples"] = 150
    assert "samples" in state
    assert state.get("samples") == 150
    assert isinstance(state.get("samples"), int)

    state["advanced_setting"] = False
    assert "advanced_setting" in state
    assert state.get("advanced_setting") == False
    assert isinstance(state.get("advanced_setting"), bool)


@pytest.mark.integration
def test_integration_remove_keys() -> None:
    """Integration test for removing keys from the session state."""
    state = st.session_state

    state["select_model"] = "LogisticRegression"
    assert "select_model" in state
    assert state.get("select_model") == "LogisticRegression"
    assert isinstance(state.get("select_model"), str)

    state.clear()
    assert "select_model" not in state
