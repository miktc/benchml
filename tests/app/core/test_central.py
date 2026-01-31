from app.core.central import CENTRAL


def test_CENTRAL() -> None:
    """Test that variables in the `CENTRAL` class have the correct type."""
    test_class = CENTRAL(
        select_model="LogisticRegression", advanced_setting=True, samples=150
    )
    assert isinstance(test_class["select_model"], str)
    assert isinstance(test_class["advanced_setting"], bool)
    assert isinstance(test_class["samples"], int)

    test_state = {"test": 10}
    assert test_state.get("test") == 10
    assert isinstance(test_state.get("test"), int)
