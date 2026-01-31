import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def run_tab_feature_comparison() -> None:
    """Renders the UI for the Feature Comparison tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    hold_header = st.empty()
    hold_header.subheader(body="Feature Comparison", divider="gray", width="content")

    df = state.get("X_features")
    configure = state.get("configure")
    add_row_noise = state.get("add_row_noise")
    samples = state.get("samples")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        if df is None or add_row_noise is None or samples is None:
            st.error(
                body="Session state keys must not be None to render the 'Feature Comparison' tab!"
            )
            return

        hold_header.empty()

        col1, col2 = st.columns(spec=[1, 0.3], vertical_alignment="bottom")
        col1.subheader(body="Feature Comparison", divider="gray", width="content")
        toggle = col2.toggle(label="Scatter | Line")

        select_feature1 = st.selectbox(label="Feature 1:", options=[col for col in df])
        select_feature2 = st.selectbox(label="Feature 2:", options=[col for col in df])

        if toggle:
            st.line_chart(
                data=df[: samples + add_row_noise][[select_feature1, select_feature2]],
                x_label="Sample Number",
                y_label="Value",
            )
        else:
            st.scatter_chart(
                data=df[: samples + add_row_noise][[select_feature1, select_feature2]],
                x_label="Sample Number",
                y_label="Value",
            )
