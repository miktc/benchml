import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def svr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `SVR` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    SVR_kernel = st.text_input(
        label="Kernel:",
        value="rbf,linear,poly,sigmoid",
        placeholder="Example: rbf,linear,poly,sigmoid",
    )
    SVR_kernel_list = parse_values(param=SVR_kernel)

    SVR_gamma = st.text_input(
        label="Gamma:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    SVR_gamma_list = parse_values(
        param=SVR_gamma, convert=True, param_type=float, min_val=0
    )

    SVR_C = st.text_input(
        label="C:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    SVR_C_list = parse_values(param=SVR_C, convert=True, param_type=float, min_val=0)

    SVR_epsilon = st.text_input(
        label="Epsilon:", value="0.1,0.5,1.0", placeholder="Example: 0.1,0.5,1.0"
    )
    SVR_epsilon_list = parse_values(
        param=SVR_epsilon, convert=True, param_type=float, min_val=0
    )

    SVR_grid = {
        "kernel": SVR_kernel_list,
        "gamma": SVR_gamma_list,
        "C": SVR_C_list,
        "epsilon": SVR_epsilon_list,
    }
    state["param_grid"] = SVR_grid

    if state.get("display_json"):
        st.json(SVR_grid, expanded=1)

    return SVR_grid
