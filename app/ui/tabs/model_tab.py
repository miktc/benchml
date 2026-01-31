import streamlit as st

from app.core.central import CENTRAL
from app.core.clean_keys import clean_keys
from typing import cast


def run_tab_model() -> None:
    """Renders the UI for the Model tab."""
    state = cast(CENTRAL, st.session_state)

    st.subheader(body="Model", divider="gray", width="content")
    model_type_select = st.selectbox(
        label="Select Model Type:", options=["Classification", "Regression"]
    )
    state["model_type_select"] = model_type_select

    model_choices_clf = [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "LinearSVC",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "MLPClassifier",
        "KNeighborsClassifier",
    ]
    model_choices_rg = [
        "LinearRegression",
        "Ridge",
        "DecisionTreeRegressor",
        "SVR",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "MLPRegressor",
        "KNeighborsRegressor",
    ]

    if model_type_select == "Classification":
        select_model = st.selectbox(label="Select Model:", options=model_choices_clf)
        state["select_model"] = select_model
    else:
        select_model = st.selectbox(label="Select Model:", options=model_choices_rg)
        state["select_model"] = select_model

    # Remove keys to clean the session state
    if state.get("update_config") != state.get("original_config"):
        clean_keys()
