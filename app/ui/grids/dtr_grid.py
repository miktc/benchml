import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def dtr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `DecisionTreeRegressor` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    DTR_criterion = st.text_input(
        label="Criterion:",
        value="squared_error,friedman_mse,absolute_error,poisson",
        placeholder="Example: squared_error,friedman_mse,absolute_error,poisson",
    )
    DTR_criterion_list = parse_values(param=DTR_criterion)

    DTR_max_depth = st.text_input(
        label="Max Depth:", value="1,2,3", placeholder="Example: 1,2,3"
    )
    DTR_max_depth_list = parse_values(
        param=DTR_max_depth, convert=True, param_type=int, min_val=1
    )

    DTR_min_samples_split = st.text_input(
        label="Min Samples Split:", value="2,4,6", placeholder="Example: 2,4,6"
    )
    DTR_min_samples_split_list = parse_values(
        param=DTR_min_samples_split, convert=True, param_type=int, min_val=2
    )

    DTR_grid = {
        "criterion": DTR_criterion_list,
        "max_depth": DTR_max_depth_list,
        "min_samples_split": DTR_min_samples_split_list,
    }
    state["param_grid"] = DTR_grid

    if state.get("display_json"):
        st.json(DTR_grid, expanded=1)

    return DTR_grid
