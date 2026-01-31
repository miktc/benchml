from app.utils.env import running_locally


def test_running_locally_true(monkeypatch) -> None:
    """Test that the `running_locally` function detects the local environment.

    Args:
        monkeypatch (MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "0")
    assert running_locally() is True


def test_running_locally_false(monkeypatch) -> None:
    """Test that the `running_locally` function detects the Streamlit cloud environment.

    Args:
        monkeypatch (MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "1")
    assert running_locally() is False
