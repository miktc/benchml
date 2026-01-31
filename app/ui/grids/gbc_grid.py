import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def gbc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `GradientBoostingClassifier` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    GBC_learning_rate = st.text_input(
        label="Learning Rate:", value="0.1,0.2,0.3", placeholder="Example: 0.1,0.2,0.3"
    )
    GBC_learning_rate_list = parse_values(
        param=GBC_learning_rate, convert=True, param_type=float, min_val=0
    )

    GBC_n_estimators = st.text_input(
        label="N Estimators:", value="100,200,300", placeholder="Example: 100,200,300"
    )
    GBC_n_estimators_list = parse_values(
        param=GBC_n_estimators, convert=True, param_type=int, min_val=1
    )

    GBC_max_depth = st.text_input(
        label="Max Depth:", value="3,5", placeholder="Example: 3,5"
    )
    GBC_max_depth_list = parse_values(
        param=GBC_max_depth, convert=True, param_type=int, min_val=1
    )

    GBC_subsample = st.text_input(
        label="Subsample:", value="0.1,0.2,0.3", placeholder="Example: 0.1,0.2,0.3"
    )
    GBC_subsample_list = parse_values(
        param=GBC_subsample, convert=True, param_type=float, min_val=0
    )

    GBC_grid = {
        "learning_rate": GBC_learning_rate_list,
        "n_estimators": GBC_n_estimators_list,
        "max_depth": GBC_max_depth_list,
        "subsample": GBC_subsample_list,
    }
    state["param_grid"] = GBC_grid

    if state.get("display_json"):
        st.json(GBC_grid, expanded=1)

    return GBC_grid
