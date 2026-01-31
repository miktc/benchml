import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def rfc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `RandomForestClassifier` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    RFC_n_estimators = st.text_input(
        label="N Estimators:", value="100,200,300", placeholder="Example: 100,200,300"
    )
    RFC_n_estimators_list = parse_values(
        param=RFC_n_estimators, convert=True, param_type=int, min_val=1
    )

    RFC_max_depth = st.text_input(
        label="Max Depth:", value="1,2,3", placeholder="Example: 1,2,3"
    )
    RFC_max_depth_list = parse_values(
        param=RFC_max_depth, convert=True, param_type=int, min_val=1
    )

    RFC_min_samples_split = st.text_input(
        label="Min Samples Split:", value="2,4,6", placeholder="Example: 2,4,6"
    )
    RFC_min_samples_split_list = parse_values(
        param=RFC_min_samples_split, convert=True, param_type=int, min_val=2
    )

    RFC_grid = {
        "n_estimators": RFC_n_estimators_list,
        "max_depth": RFC_max_depth_list,
        "min_samples_split": RFC_min_samples_split_list,
    }
    state["param_grid"] = RFC_grid

    if state.get("display_json"):
        st.json(RFC_grid, expanded=1)

    return RFC_grid
