import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def dtc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `DecisionTreeClassifier` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    DTC_criterion = st.text_input(
        label="Criterion:",
        value="gini,entropy,log_loss",
        placeholder="Example: gini,entropy,log_loss",
    )
    DTC_criterion_list = parse_values(param=DTC_criterion)

    DTC_max_depth = st.text_input(
        label="Max Depth:", value="1,2,3", placeholder="Example: 1,2,3"
    )
    DTC_max_depth_list = parse_values(
        param=DTC_max_depth, convert=True, param_type=int, min_val=1
    )

    DTC_min_samples_split = st.text_input(
        label="Min Samples Split:",
        value="2,4,6,8,10",
        placeholder="Example: 2,4,6,8,10",
    )
    DTC_min_samples_split_list = parse_values(
        param=DTC_min_samples_split, convert=True, param_type=int, min_val=2
    )

    DTC_grid = {
        "criterion": DTC_criterion_list,
        "max_depth": DTC_max_depth_list,
        "min_samples_split": DTC_min_samples_split_list,
    }
    state["param_grid"] = DTC_grid

    if state.get("display_json"):
        st.json(DTC_grid, expanded=1)

    return DTC_grid
