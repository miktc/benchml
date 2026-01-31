import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def linearsvc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `LinearSVC` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    LSVC_penalty = st.text_input(
        label="Penalty:", value="l1,l2", placeholder="Example: l1,l2"
    )
    LSVC_penalty_list = parse_values(param=LSVC_penalty)

    LSVC_loss = st.text_input(
        label="Loss:",
        value="squared_hinge,hinge",
        placeholder="Example: squared_hinge,hinge",
    )
    LSVC_loss_list = parse_values(param=LSVC_loss)

    LSVC_C = st.text_input(
        label="C:", value="0.01,0.1,1", placeholder="Example: 0.01,0.1,1"
    )
    LSVC_C_list = parse_values(param=LSVC_C, convert=True, param_type=float, min_val=0)

    LSVC_max_iter = st.text_input(
        label="Max Iter:", value="1000,1500,2000", placeholder="Example: 1000,1500,2000"
    )
    LSVC_max_iter_list = parse_values(
        param=LSVC_max_iter, convert=True, param_type=int, min_val=1
    )

    LSVC_C_grid = {
        "penalty": LSVC_penalty_list,
        "loss": LSVC_loss_list,
        "C": LSVC_C_list,
        "max_iter": LSVC_max_iter_list,
    }
    state["param_grid"] = LSVC_C_grid

    if state.get("display_json"):
        st.json(LSVC_C_grid, expanded=1)

    return LSVC_C_grid
