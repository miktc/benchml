import json
import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def run_tab_export() -> None:
    """Renders the UI for the Export tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="Export", divider="gray", width="content")
    configure = state.get("configure")
    advanced_setting = state.get("advanced_setting")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        display_data = st.empty()
        display_data.markdown(body="#### Data")
        file_error = False
        if state.get("sql_toggle"):
            try:
                with open("BenchML-DB.db", "rb") as file:
                    st.download_button(
                        label="Download Database",
                        data=file.read(),
                        file_name="BenchML-DB.db",
                        mime="application/octet-stream",
                        width=200,
                    )
            except FileNotFoundError:
                file_error = True

        try:
            with open("dataset.csv", "r") as file:
                st.download_button(
                    label="Download Dataset",
                    data=file.read().encode("utf-8"),
                    file_name="BenchML-Dataset.csv",
                    width=200,
                )
        except FileNotFoundError:
            if file_error:
                display_data.empty()

        st.markdown(body="#### Metrics")
        all_metrics = state.get("all_metrics")

        if all_metrics is None:
            st.error(body="No metrics available to export!")
            return

        metrics_data = json.dumps(all_metrics, indent=4)
        st.download_button(
            label="Download Metrics",
            data=metrics_data,
            file_name="BenchML-Metrics.json",
            width=200,
        )

        if advanced_setting is None:
            st.error(
                body="The session state key 'advanced_setting' must not be None to export!"
            )
            return

        if advanced_setting:
            st.markdown(body="#### Parameters")
            best_params = state.get("best_params")

            if best_params is None:
                st.error(body="No parameters available to export!")
                return

            param_data = json.dumps(best_params, indent=4)
            st.download_button(
                label="Download Parameters",
                data=param_data,
                file_name=f"BenchML-Params.json",
                width=200,
            )

            st.markdown(body="#### Parameter Grid")
            param_grid = state.get("param_grid")
            if param_grid is None:
                st.error(body="No parameter grid available to export!")
                return

            param_grid_data = json.dumps(param_grid, indent=4)
            st.download_button(
                label="Download Parameter Grid",
                data=param_grid_data,
                file_name=f"BenchML-Param_Grid.json",
                width=200,
            )
