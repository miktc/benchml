import streamlit as st

from typing import Callable, Optional


def try_except(
    *funcs: Callable, info: bool = False, message: Optional[str] = None
) -> None:
    """Calls given functions inside a try-except block.

    Args:
        *funcs (Callable): Functions to call.
        info (bool, optional): True to display a Streamlit info message after an exception. Defaults to False.
        message (str, optional): Text to display in the info message when the exception occurs. Defaults to None.

    Raises:
        Exception: If calling the function raises an error.
    """
    for func in funcs:
        try:
            func()
        except Exception:
            if info:
                st.info(message)
            else:
                pass
