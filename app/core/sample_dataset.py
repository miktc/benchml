import sklearn.datasets
import streamlit as st

from app.core.central import CENTRAL
from typing import Any, cast


def sample_dataset() -> Any:
    """Fetches a sample dataset based on the selection in the Streamlit session state.

    Returns:
        Any: The selected sample dataset or None if no valid selection is made.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    sample_dataset = state.get("select_sample_dataset")

    if sample_dataset is None:
        st.error(body="The sample dataset must not be None!")
        return None

    if sample_dataset == "Iris":
        return sklearn.datasets.load_iris()
    elif sample_dataset == "Breast Cancer":
        return sklearn.datasets.load_breast_cancer()
    elif sample_dataset == "Diabetes":
        return sklearn.datasets.load_diabetes()
    elif sample_dataset == "Digits":
        return sklearn.datasets.load_digits()
    elif sample_dataset == "Wine":
        return sklearn.datasets.load_wine()
    else:
        st.error(body="Received an unexpected name for sample data!")
        return None
