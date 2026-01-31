import pandas as pd
import streamlit as st
import warnings

from app.core.central import CENTRAL
from app.core.get_param_grid import get_param_grid
from app.core.ml_models import models
from app.core.synthetic_data import add_feature_cols, add_rows
from app.services.mlflow_log import run_mlflow
from app.services.sql_log import SQL_Storage
from app.utils.env import running_locally
from datetime import datetime
from typing import cast


def run_tab_configure() -> None:
    """Renders the UI for the Configure tab.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    if "config_duration" not in st.session_state:
        st.session_state["config_duration"] = 0

    if "configure" not in st.session_state:
        st.session_state["configure"] = False

    st.subheader(body="Configure", divider="gray", width="content")
    df = state.get("df")
    model_type_select = state.get("model_type_select")
    select_model = state.get("select_model")
    select_sample_dataset = state.get("select_sample_dataset")
    uploaded_file = state.get("uploaded_file")

    if (
        df is None
        and model_type_select is None
        and select_model is None
        and select_sample_dataset is None
    ):
        st.error(body="Session state keys must not be None when configuring!")
        return

    if not uploaded_file and not state.get("render_after_upload"):
        st.info(
            body=f"File '{state.get('uploaded_file_name', '')}' failed to upload: Upload a valid CSV file in the 'Select Data' tab!"
        )
        return

    df = cast(pd.DataFrame, st.session_state.get("df"))
    model_type_select = cast(str, st.session_state.get("model_type_select"))
    select_model = cast(str, st.session_state.get("select_model"))
    select_sample_dataset = cast(str, st.session_state.get("select_sample_dataset"))
    uploaded_file = cast(bool, st.session_state.get("uploaded_file"))

    col1, _ = st.columns(spec=(2))
    st.divider()
    st.markdown(body="#### **Settings**")
    advanced_setting = st.toggle(label="Enable Advanced")
    sql_toggle = st.toggle(label="Enable SQL Storage")

    if sql_toggle:
        state["sql_toggle"] = True

    if running_locally():
        mlflow_toggle = st.toggle(label="Enable MLflow")
        if mlflow_toggle:
            state["mlflow_toggle"] = True
    else:
        mlflow_toggle = st.toggle(label="Enable MLflow", disabled=True)
        st.caption(body="MLflow is available for local use only.")

    st.divider()

    state["advanced_setting"] = False
    st.markdown(body="#### **Data**")

    samples = st.slider(
        label="Original Samples:",
        min_value=1,
        max_value=len(df),
        value=len(df),
        help="Select the total number of original samples.",
    )
    state["samples"] = samples
    state["df"] = df[:samples]

    if samples == 1:
        st.session_state["configure"] = False
        st.warning(body="More samples are required!")
        return
    else:
        add_row_noise = st.slider(
            label="Add Rows:",
            min_value=0,
            max_value=200,
            help="Add synthetic rows to both the features and target within the dataset.",
        )
        state["add_row_noise"] = add_row_noise

        # Add rows after the target value selection in advanced settings
        if not advanced_setting:
            add_rows()

        add_feature_noise = st.slider(
            label="Add Features:",
            min_value=0,
            max_value=50,
            help="Add synthetic features to the dataset.",
        )
        state["add_feature_noise"] = add_feature_noise
        add_feature_cols()

        test_size = st.slider(
            label="Test Size:",
            min_value=1,
            max_value=100,
            value=25,
            help="Select the percentage used for testing data.",
        )
        state["test_size"] = test_size

        target = st.selectbox(
            label="Target Variable:",
            options=[col for col in df],
            help="Select the target variable for the dataset.",
        )
        state["target"] = str(target)

        # Advanced setting options
        if advanced_setting:
            state["advanced_setting"] = True

            select_target_value = st.selectbox(
                label="Target Variable Value:",
                options=["No", "Yes"],
                help="Select 'Yes' to fix the value of the target variable when rows are added.",
            )
            state["select_target_value"] = select_target_value

            try:
                if select_target_value == "Yes":
                    target_value = st.slider(
                        label="Target Variable Value:",
                        min_value=df[target].min(),
                        max_value=df[target].max(),
                    )
                    state["target_value"] = target_value
                    add_rows()
                else:
                    add_rows()
            except Exception:
                st.warning(
                    body="Check the target variable selection! The default value is 0."
                )
                target_value = 0
                state["target_value"] = target_value
                add_rows()

            st.divider()

            st.markdown(body="#### **Search**")
            search_method = st.selectbox(
                label="Select Search Method:",
                options=["GridSearchCV", "RandomizedSearchCV"],
            )
            state["search_method"] = search_method

            if search_method == "RandomizedSearchCV":
                select_random_state = st.number_input(
                    label="Random State:",
                    min_value=0,
                    max_value=(2**32) - 1,
                    value=42,
                )
                state["select_random_state"] = select_random_state
            else:
                select_random_state = 42

            select_cv = st.number_input(label="CV:", min_value=1, max_value=10, value=3)
            state["select_cv"] = select_cv

            st.divider()

            grid_col1, grid_col2 = st.columns(spec=[1, 0.25], gap="small")
            grid_col1.markdown(body="#### **Parameter Grid**")
            display_json = grid_col2.toggle(label="Display JSON")
            state["display_json"] = display_json

            param_grid = get_param_grid()
            state["param_grid"] = param_grid
            confirm_settings = st.button(label="Configure", width=180)

            if select_target_value == "Yes":
                # Snapshot of settings with a set target value
                state["original_config"] = {
                    "model_type_select": model_type_select,
                    "select_model": select_model,
                    "select_sample_dataset": select_sample_dataset,
                    "uploaded_file": uploaded_file,
                    "mlflow_toggle": mlflow_toggle,
                    "advanced_setting": advanced_setting,
                    "sql_toggle": sql_toggle,
                    "samples": samples,
                    "add_row_noise": add_row_noise,
                    "add_feature_noise": add_feature_noise,
                    "test_size": test_size,
                    "target": target,
                    "select_target_value": select_target_value,
                    "target_value": target_value,
                    "select_random_state": select_random_state,
                    "select_cv": select_cv,
                    "search_method": search_method,
                    "param_grid": param_grid,
                }
            else:
                state["original_config"] = {
                    "model_type_select": model_type_select,
                    "select_model": select_model,
                    "select_sample_dataset": select_sample_dataset,
                    "uploaded_file": uploaded_file,
                    "mlflow_toggle": mlflow_toggle,
                    "advanced_setting": advanced_setting,
                    "sql_toggle": sql_toggle,
                    "samples": samples,
                    "add_row_noise": add_row_noise,
                    "add_feature_noise": add_feature_noise,
                    "test_size": test_size,
                    "target": target,
                    "select_target_value": select_target_value,
                    "select_random_state": select_random_state,
                    "select_cv": select_cv,
                    "search_method": search_method,
                    "param_grid": param_grid,
                }
        else:
            # Basic settings
            confirm_settings = st.button(label="Configure", width=180)

            state["original_config"] = {
                "model_type_select": model_type_select,
                "select_model": select_model,
                "select_sample_dataset": select_sample_dataset,
                "uploaded_file": uploaded_file,
                "mlflow_toggle": mlflow_toggle,
                "advanced_setting": advanced_setting,
                "sql_toggle": sql_toggle,
                "samples": samples,
                "add_row_noise": add_row_noise,
                "add_feature_noise": add_feature_noise,
                "test_size": test_size,
                "target": target,
            }

    if samples != 1:
        try:
            row_noise_percent = round(
                (add_row_noise) / (samples + add_row_noise) * 100, 2
            )
            state["row_noise_percent"] = row_noise_percent

            feature_noise_percent = round(
                ((add_feature_noise) / (len(df.columns) - 1 + add_feature_noise)) * 100,
                2,
            )
            state["feature_noise_percent"] = feature_noise_percent

            col1.markdown(
                f"""
                ##### Row Noise: {row_noise_percent}%  
                ##### Feature Noise: {feature_noise_percent}%
            """
            )
        except ZeroDivisionError:
            row_noise_percent = round(
                (add_row_noise) / (samples + add_row_noise) * 100, 2
            )
            state["row_noise_percent"] = row_noise_percent

            feature_noise_percent = round(
                (((add_feature_noise) / (len(df.columns) + add_feature_noise))) * 100,
                2,
            )
            state["feature_noise_percent"] = feature_noise_percent

            col1.markdown(
                f"""
                ##### Row Noise: {row_noise_percent}%
                ##### Feature Noise: {feature_noise_percent}%
            """
            )

    progress_bar = st.empty()
    invalid_entries = st.empty()

    if confirm_settings:
        start_time = datetime.now()
        progress_bar.progress(value=0, text="Configuring...")

        warnings.filterwarnings("ignore")

        state["configure"] = True
        original_config = state.get("original_config")

        if original_config is not None:
            state["update_config"] = original_config.copy()

        if advanced_setting:
            if not state.get("valid_entry"):
                progress_bar.empty()
                state["configure"] = False
                invalid_entries.warning(body="Invalid Entries!")
                st.toast(body="Configuration Failed!", icon="⚠️")
            else:
                invalid_entries.empty()
                state["configure"] = True
                try:
                    models()
                    progress_bar.progress(value=25, text="Configuring...")
                except Exception:
                    progress_bar.empty()
                    state["configure"] = False
                    st.error(body=f"Failed to configure {select_model}!")

                if state.get("configure"):
                    end_time = datetime.now()
                    duration = end_time - start_time
                    state["config_duration"] = duration.total_seconds()
                    state["time_created"] = str(end_time)

                    if sql_toggle:
                        progress_bar.progress(value=50, text="Configuring...")
                        SQL_Storage()

                    if mlflow_toggle:
                        progress_bar.progress(value=75, text="Configuring...")
                        mlflow_conn = run_mlflow()
                        if not mlflow_conn:
                            st.warning(
                                body="MLflow connection failed. Configuration completed without MLflow logging!"
                            )
                            st.toast(
                                body="MLflow connection failed. Configuring without MLflow!",
                                icon="⚠️",
                            )

                    progress_bar.progress(value=100, text="Completed!")
                    st.toast(body="Configuration Complete!", icon="✅")
                else:
                    progress_bar.empty()
                    st.toast(body="Configuration Failed!", icon="⚠️")
        else:
            # Standard setting options
            state["configure"] = True
            try:
                models()
                progress_bar.progress(value=25, text="Configuring...")
            except Exception:
                progress_bar.empty()
                state["configure"] = False
                st.error(body=f"Failed to configure {select_model}!")

            if state.get("configure"):
                end_time = datetime.now()
                duration = end_time - start_time
                state["config_duration"] = duration.total_seconds()
                state["time_created"] = str(end_time)

                if sql_toggle:
                    progress_bar.progress(value=50, text="Configuring...")
                    SQL_Storage()

                if mlflow_toggle:
                    progress_bar.progress(value=75, text="Configuring...")
                    mlflow_conn = run_mlflow()
                    if not mlflow_conn:
                        st.warning(
                            body="MLflow connection failed. Configuration completed without MLflow logging!"
                        )
                        st.toast(
                            body="MLflow connection failed. Configuring without MLflow!",
                            icon="⚠️",
                        )

                progress_bar.progress(value=100, text="Completed!")
                st.toast(body="Configuration Complete!", icon="✅")
            else:
                progress_bar.empty()
                st.toast(body="Configuration Failed!", icon="⚠️")
