import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import parse_values
from typing import Any, cast


def knc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `KNeighborsClassifier` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    KNC_n_neighbors = st.text_input(
        label="N Neighbors:", value="5,10,15", placeholder="Example: 5,10,15"
    )
    KNC_n_neighbors_list = parse_values(
        param=KNC_n_neighbors, convert=True, param_type=int, min_val=1
    )

    KNC_weights = st.text_input(
        label="Weights:",
        value="uniform,distance",
        placeholder="Example: uniform,distance",
    )
    KNC_weights_list = parse_values(param=KNC_weights)

    KNC_metric = st.text_input(
        label="Metric:", value="minkowski", placeholder="Example: minkowski"
    )
    KNC_metric_list = parse_values(param=KNC_metric)

    KNC_grid = {
        "n_neighbors": KNC_n_neighbors_list,
        "weights": KNC_weights_list,
        "metric": KNC_metric_list,
    }
    state["param_grid"] = KNC_grid

    if state.get("display_json"):
        st.json(KNC_grid, expanded=1)

    return KNC_grid
