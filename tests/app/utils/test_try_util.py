from app.utils.try_util import try_except


def test_try_except() -> None:
    """Test that the function `try_except` does not raise any errors when calling other functions."""

    def raise_value_error() -> None:
        """Test function that raises a ValueError."""
        raise ValueError()

    def raise_type_error() -> None:
        """Test function that raises a TypeError."""
        raise TypeError()

    def raise_exception() -> None:
        """Test function that raises an Exception."""
        raise Exception()

    try_except(raise_value_error)
    try_except(raise_type_error)
    try_except(raise_exception)
