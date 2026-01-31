import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def rfr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `RandomForestRegressor` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    RFR_n_estimators = st.text_input(
        label="N Estimators:", value="100,200,300", placeholder="Example: 100,200,300"
    )
    RFR_n_estimators_list = parse_values(
        param=RFR_n_estimators, convert=True, param_type=int, min_val=1
    )

    RFR_max_depth = st.text_input(
        label="Max Depth:", value="1,5,10", placeholder="Example: 1,5,10"
    )
    RFR_max_depth_list = parse_values(
        param=RFR_max_depth, convert=True, param_type=int, min_val=1
    )

    RFR_min_samples_split = st.text_input(
        label="Min Samples Split:", value="2,4,6", placeholder="Example: 2,4,6"
    )
    RFR_min_samples_split_list = parse_values(
        param=RFR_min_samples_split, convert=True, param_type=int, min_val=2
    )

    RFR_bootstrap = st.selectbox(
        label="Bootstrap:", options=["True,False", "True", "False"]
    )

    if RFR_bootstrap == "True":
        RFR_bootstrap_list = [True]
    elif RFR_bootstrap == "False":
        RFR_bootstrap_list = [False]
    else:
        RFR_bootstrap_list = [True, False]

    RFR_grid = {
        "n_estimators": RFR_n_estimators_list,
        "max_depth": RFR_max_depth_list,
        "min_samples_split": RFR_min_samples_split_list,
        "bootstrap": RFR_bootstrap_list,
    }
    state["param_grid"] = RFR_grid

    if state.get("display_json"):
        st.json(RFR_grid, expanded=1)

    return RFR_grid
