import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def ridge_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `Ridge` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    RIDGE_alpha = st.text_input(
        label="Alpha:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    RIDGE_alpha_list = parse_values(
        param=RIDGE_alpha, convert=True, param_type=float, min_val=0
    )

    RIDGE_solver = st.text_input(
        label="Solver:",
        value="svd,cholesky,lsqr,sparse_cg,sag,saga,lbfgs",
        placeholder="Example: svd,cholesky,lsqr,sparse_cg,sag,saga,lbfgs",
    )
    RIDGE_solver_list = parse_values(param=RIDGE_solver)

    RIDGE_max_iter = st.text_input(
        label="Max Iter:", value="1,2,3", placeholder="Example: 1,2,3"
    )
    RIDGE_max_iter_list = parse_values(
        param=RIDGE_max_iter, convert=True, param_type=int, min_val=1
    )

    RIDGE_grid = {
        "alpha": RIDGE_alpha_list,
        "solver": RIDGE_solver_list,
        "max_iter": RIDGE_max_iter_list,
    }
    state["param_grid"] = RIDGE_grid

    if state.get("display_json"):
        st.json(RIDGE_grid, expanded=1)

    return RIDGE_grid
