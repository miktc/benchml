import pandas as pd
import streamlit as st

from app.core.central import CENTRAL
from app.core.sample_dataset import sample_dataset
from typing import cast


def run_tab_select_data() -> None:
    """Renders the UI for the Select Data tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = ""

    st.subheader(body="Select Data", divider="gray", width="content")
    model_type_select = state.get("model_type_select")

    if model_type_select is None:
        st.error(body="Session state key 'model_type_select' must not be None!")
        return

    if model_type_select == "Classification":
        select_sample_dataset = st.selectbox(
            label="Select Sample Data:",
            options=["Iris", "Breast Cancer", "Digits", "Wine"],
            help="Sample datasets are used when a file is not uploaded.",
        )
    else:
        select_sample_dataset = st.selectbox(
            label="Select Sample Data:", options=["Diabetes"]
        )
    state["select_sample_dataset"] = select_sample_dataset

    uploaded_file = st.file_uploader(
        label="Choose a file:",
        type=["csv"],
        accept_multiple_files=False,
        help="Valid CSV files contain only numeric values and at least two columns.",
    )

    if uploaded_file and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) == 1:
                st.error(body=f"{uploaded_file.name} contains only one column!")
                state["uploaded_file"] = False
                state["render_after_upload"] = False
                state["uploaded_file_name"] = uploaded_file.name
            elif len(df.values) == 0:
                st.error(body=f"{uploaded_file.name} contains no data!")
                state["uploaded_file"] = False
                state["render_after_upload"] = False
                state["uploaded_file_name"] = uploaded_file.name
            elif len(df) == 1:
                st.error(body=f"{uploaded_file.name} contains only one row!")
                state["uploaded_file"] = False
                state["render_after_upload"] = False
                state["uploaded_file_name"] = uploaded_file.name
            elif [col for col in df.columns if df[col].dropna().empty]:
                st.error(body=f"{uploaded_file.name} contains empty columns!")
                state["uploaded_file"] = False
                state["render_after_upload"] = False
                state["uploaded_file_name"] = uploaded_file.name
            elif not df.apply(pd.to_numeric, errors="coerce").notna().all(axis=1).all():
                st.error(body=f"{uploaded_file.name} contains non-numeric values!")
                state["uploaded_file"] = False
                state["render_after_upload"] = False
                state["uploaded_file_name"] = uploaded_file.name
            else:
                state["uploaded_file"] = True
                state["uploaded_file_name"] = uploaded_file.name
                state["df"] = df
                st.success(body=f"File '{uploaded_file.name}' successfully uploaded!")
        except ValueError as e:
            st.error(body=f"File '{uploaded_file.name}' failed to upload: {e}")
            state["uploaded_file"] = False
            state["render_after_upload"] = False
            state["uploaded_file_name"] = uploaded_file.name

    if select_sample_dataset and not uploaded_file:
        state["uploaded_file"] = False
        state["render_after_upload"] = True

        data = sample_dataset()
        sample_data = data["data"]
        sample_feature_names = data["feature_names"]
        sample_target = data["target"]

        df = pd.DataFrame(data=sample_data, columns=sample_feature_names)

        # Add target to sample dataframe
        df["target"] = sample_target

        # Keys for sample data
        state["df"] = df
        state["sample_data"] = sample_data
        state["sample_feature_names"] = sample_feature_names
        state["sample_target"] = sample_target
