import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def lr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `LogisticRegression` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    LR_C = st.text_input(
        label="C:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    LR_C_list = parse_values(param=LR_C, convert=True, param_type=float, min_val=0)

    LR_penalty = st.text_input(
        label="Penalty:",
        value="l2,l1,elasticnet",
        placeholder="Example: l2,l1,elasticnet",
    )
    LR_penalty_list = parse_values(param=LR_penalty)

    LR_solver = st.text_input(
        label="Solver:",
        value="lbfgs,newton-cg,newton-cholesky,sag,saga",
        placeholder="Example: lbfgs,newton-cg,newton-cholesky,sag,saga",
    )
    LR_solver_list = parse_values(param=LR_solver)

    LR_max_iter = st.text_input(
        label="Max Iter:", value="100,200,300", placeholder="Example: 100,200,300"
    )
    LR_max_iter_list = parse_values(
        param=LR_max_iter, convert=True, param_type=int, min_val=1
    )

    LR_C_grid = {
        "C": LR_C_list,
        "penalty": LR_penalty_list,
        "solver": LR_solver_list,
        "max_iter": LR_max_iter_list,
    }
    state["param_grid"] = LR_C_grid

    if state.get("display_json"):
        st.json(LR_C_grid, expanded=1)

    return LR_C_grid
