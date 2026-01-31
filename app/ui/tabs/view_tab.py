import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def run_tab_view() -> None:
    """Renders the UI for the View tab.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="View", divider="gray", width="content")
    df = state.get("df")

    if not state.get("configure"):
        st.info(body="Configuration is required!")
        return

    # Not configured when modifying the Model tab or Select Data tab
    if state.get("update_config") != state.get("original_config"):
        state["configure"] = False
        st.info(body="Configuration is required!")
    else:
        if df is None:
            st.error(
                body="Session state key 'df' must not be None to render the 'View' tab!"
            )
            return

        st.dataframe(data=df)
