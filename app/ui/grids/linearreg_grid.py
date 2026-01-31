import streamlit as st

from app.core.central import CENTRAL
from typing import Any, cast


def linearreg_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `LinearRegression` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    LR_intercept = st.selectbox(
        label="Fit Intercept:", options=["True,False", "True", "False"]
    )
    if LR_intercept == "True":
        LR_intercept_list = [True]
    elif LR_intercept == "False":
        LR_intercept_list = [False]
    else:
        LR_intercept_list = [True, False]

    LR_copy_X = st.selectbox(label="Copy_X:", options=["True,False", "True", "False"])
    if LR_copy_X == "True":
        LR_copy_X_list = [True]
    elif LR_copy_X == "False":
        LR_copy_X_list = [False]
    else:
        LR_copy_X_list = [True, False]

    LR_grid = {"fit_intercept": LR_intercept_list, "copy_X": LR_copy_X_list}
    state["param_grid"] = LR_grid

    if state.get("display_json"):
        st.json(LR_grid, expanded=1)

    return LR_grid
