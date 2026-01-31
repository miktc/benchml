import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def gbr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `GradientBoostingRegressor` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    GBR_learning_rate = st.text_input(
        label="Learning Rate:", value="0.01,0.1", placeholder="Example: 0.01,0.1"
    )
    GBR_learning_rate_list = parse_values(
        param=GBR_learning_rate, convert=True, param_type=float, min_val=0
    )

    GBR_n_estimators = st.text_input(
        label="N Estimators:", value="100,200,300", placeholder="Example: 100,200,300"
    )
    GBR_n_estimators_list = parse_values(
        param=GBR_n_estimators, convert=True, param_type=int, min_val=1
    )

    GBR_subsample = st.text_input(
        label="Subsample:", value="0.1,0.2,0.3", placeholder="Example: 0.1,0.2,0.3"
    )
    GBR_subsample_list = parse_values(
        param=GBR_subsample, convert=True, param_type=float, min_val=0
    )

    GBR_max_depth = st.text_input(
        label="Max Depth:", value="3,5,7", placeholder="Example: 3,5,7"
    )
    GBR_max_depth_list = parse_values(
        param=GBR_max_depth, convert=True, param_type=int, min_val=1
    )

    GBR_grid = {
        "learning_rate": GBR_learning_rate_list,
        "n_estimators": GBR_n_estimators_list,
        "subsample": GBR_subsample_list,
        "max_depth": GBR_max_depth_list,
    }
    state["param_grid"] = GBR_grid

    if state.get("display_json"):
        st.json(GBR_grid, expanded=1)

    return GBR_grid
