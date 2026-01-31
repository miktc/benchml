import app.core.visuals as visuals
import streamlit as st

from app.core.central import CENTRAL
from app.utils.try_util import try_except
from typing import cast


def run_tab_visualize() -> None:
    """Renders the UI for the Visualize tab.

    Notes:
        If the required session state keys are None or an invalid visualization is selected,
        a Streamlit error message is displayed using `st.error` and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="Visualize", divider="gray", width="content")
    configure = state.get("configure")
    select_model = state.get("select_model")
    model_type_select = state.get("model_type_select")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        # Not configured when modifying the Model tab or Select Data tab
        if state.get("update_config") != state.get("original_config"):
            state["configure"] = False
            st.info(body="Configuration is required!")
        else:
            if select_model is None or model_type_select is None:
                st.error(
                    body="Session state keys 'select_model' and 'model_type_select' must not be None!"
                )
                return

            if model_type_select == "Classification":
                select_visual = st.selectbox(
                    label="Select Visualization:",
                    options=[
                        "Confusion Matrix",
                        "ROC Curve",
                        "Precision-Recall Curve",
                        "SHAP Summary Plot",
                    ],
                )
                if select_visual == "Confusion Matrix":
                    try_except(
                        lambda: visuals.create_cm(display=True),
                        info=True,
                        message=f"The confusion matrix is currently not supported for {select_model}!",
                    )
                elif select_visual == "ROC Curve":
                    try_except(
                        lambda: visuals.create_roc_curve(display=True),
                        info=True,
                        message=f"The ROC curve is currently not supported for {select_model}!",
                    )
                elif select_visual == "Precision-Recall Curve":
                    try_except(
                        lambda: visuals.create_pr_curve(display=True),
                        info=True,
                        message=f"The precision-recall curve is currently not supported for {select_model}!",
                    )
                elif select_visual == "SHAP Summary Plot":
                    try_except(
                        lambda: visuals.create_shap(display=True),
                        info=True,
                        message=f"The SHAP summary plot is currently not supported for {select_model}!",
                    )
                else:
                    st.error(
                        body=f"Received an unexpected visualization: {select_visual}!"
                    )
                    return
            else:
                select_visual = st.selectbox(
                    label="Select Visualization:",
                    options=[
                        "Predicted vs. Actual Values Plot",
                        "Error Distribution Plot",
                        "Residual Plot",
                        "SHAP Summary Plot",
                    ],
                    help="Select a visualization to display.",
                )
                if select_visual == "Predicted vs. Actual Values Plot":
                    try_except(
                        lambda: visuals.create_predict_actual(display=True),
                        info=True,
                        message=f"The predicted vs. actual values plot is currently not supported for {select_model}!",
                    )
                elif select_visual == "Error Distribution Plot":
                    try_except(
                        lambda: visuals.create_error_distribution(display=True),
                        info=True,
                        message=f"The error distribution plot is currently not supported for {select_model}!",
                    )
                elif select_visual == "Residual Plot":
                    try_except(
                        lambda: visuals.create_residual(display=True),
                        info=True,
                        message=f"The residual plot is currently not supported for {select_model}!",
                    )
                elif select_visual == "SHAP Summary Plot":
                    try_except(
                        lambda: visuals.create_shap(display=True),
                        info=True,
                        message=f"The SHAP summary plot is currently not supported for {select_model}!",
                    )
                else:
                    st.error(
                        body=f"Received an unexpected visualization: {select_visual}!"
                    )
                    return
