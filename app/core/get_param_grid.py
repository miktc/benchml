import streamlit as st

from app.core.central import CENTRAL
from app.ui.grids import (
    dtc_grid,
    dtr_grid,
    gbc_grid,
    gbr_grid,
    knc_grid,
    knr_grid,
    linearreg_grid,
    linearsvc_grid,
    lr_grid,
    mlpc_grid,
    mlpr_grid,
    rfc_grid,
    rfr_grid,
    ridge_grid,
    svr_grid,
)
from typing import Any, cast


def get_param_grid() -> dict | dict[str, list[Any]]:
    """Retrieves the parameter grid for the selected machine learning model.

    Returns:
        dict | dict[str, list[Any]]
            The `param_grid` where the key
            is a string, and the value is a list containing any type.
            An empty dictionary is returned if the model selection is invalid.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and an empty dictionary is returned.
    """
    state = cast(CENTRAL, st.session_state)

    state["valid_entry"] = True
    select_model = state.get("select_model")

    if select_model is None:
        st.error(
            body="Invalid 'select_model' type within the session state.",
        )
        return {}

    grid_dict = {
        "LogisticRegression": lr_grid,
        "DecisionTreeClassifier": dtc_grid,
        "LinearSVC": linearsvc_grid,
        "RandomForestClassifier": rfc_grid,
        "GradientBoostingClassifier": gbc_grid,
        "MLPClassifier": mlpc_grid,
        "KNeighborsClassifier": knc_grid,
        "LinearRegression": linearreg_grid,
        "Ridge": ridge_grid,
        "DecisionTreeRegressor": dtr_grid,
        "SVR": svr_grid,
        "RandomForestRegressor": rfr_grid,
        "GradientBoostingRegressor": gbr_grid,
        "MLPRegressor": mlpr_grid,
        "KNeighborsRegressor": knr_grid,
    }

    return grid_dict[select_model]()
