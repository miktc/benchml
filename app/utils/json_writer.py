import json
import streamlit as st


def JSON_Writer() -> None:
    """Creates a JSON file of the parameter grid named `param_grid.json`.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    grid = st.session_state.get("param_grid")

    if grid is None:
        st.error(body="No parameter grid is available to write a JSON file!")
        return

    with open("param_grid.json", "w", newline="\n") as file:
        json.dump(grid, file, indent=4)
