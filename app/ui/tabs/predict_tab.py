import streamlit as st

from app.core.central import CENTRAL
from app.core.ml_models import models
from typing import Any, cast


def run_tab_predict() -> None:
    """Renders the UI for the Predict tab.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="Predict", divider="gray", width="content")
    configure = state.get("configure")
    X_features = state.get("X_features")

    if not configure:
        st.info(body="Configuration is required!")
    else:
        # Not configured when modifying the Model tab or Select Data tab
        if state.get("update_config") != state.get("original_config"):
            state["configure"] = False
            st.info(body="Configuration is required!")
        else:
            st.write("Enter values to generate a prediction.")
            val_dict: dict[str, Any] = {}
            predict_form = {}

            if X_features is None:
                st.error(body="Feature data must not be None in session state!")
                return

            with st.form(key="entry_form"):
                for val in range(len(X_features.columns)):
                    text_val = st.text_input(
                        label=f"{[col for col in X_features][val]}:"
                    )
                    val_dict.update({f"{[col for col in X_features][val]}": text_val})

                predict = st.form_submit_button(label="Predict", width=125)
                state["predict_button"] = predict

                if predict:
                    st.divider()
                    warn_msg = st.empty()
                    wait_msg = st.empty()
                    wait_msg.info(body="Generating Prediction...")

                    try:
                        for key, val in val_dict.items():
                            predict_form.update({key: [float(val)]})

                        state["predict_form"] = predict_form

                        models(mode="predict")

                        result = state.get("predict_result")
                        confidence = state.get("confidence")
                        uncertainty = state.get("uncertainty")
                        warn_msg.empty()

                        if state.get("model_type_select") == "Classification":
                            wait_msg.empty()
                            st.markdown(
                                body=f"""
                                #### Prediction: {result}
                                #### Confidence: {confidence}
                            """
                            )
                        else:
                            wait_msg.empty()
                            st.markdown(
                                body=f"""
                                #### Prediction: {result}
                                #### Uncertainty: {uncertainty}
                            """
                            )

                        st.toast(body="Prediction Ready!", icon="✅")
                    except ValueError:
                        wait_msg.empty()
                        warn_msg.warning(body="Enter Numeric Values!")
                        st.toast(body="Prediction Failed!", icon="⚠️")
