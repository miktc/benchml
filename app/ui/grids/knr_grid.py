import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def knr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `KNeighborsRegressor` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    KNR_n_neighbors = st.text_input(
        label="N Neighbors:", value="5,10,15", placeholder="Example: 5,10,15"
    )
    KNR_n_neighbors_list = parse_values(
        param=KNR_n_neighbors, convert=True, param_type=int, min_val=1
    )

    KNR_weights = st.text_input(
        label="Weights:",
        value="uniform,distance",
        placeholder="Example: uniform,distance",
    )
    KNR_weights_list = parse_values(param=KNR_weights)

    KNR_grid = {"n_neighbors": KNR_n_neighbors_list, "weights": KNR_weights_list}
    state["param_grid"] = KNR_grid

    if state.get("display_json"):
        st.json(KNR_grid, expanded=1)

    return KNR_grid
