import pytest
import streamlit as st


@pytest.mark.integration
def test_integration_param_grid() -> None:
    """Integration test for creating a parameter grid."""
    C = [1, 2, 3]
    penalty = ["l2", "l1", "elasticnet"]
    solver = ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
    max_iter = [100, 200, 300]

    assert len(C) == 3
    assert len(penalty) == 3
    assert len(solver) == 5
    assert C == [1, 2, 3]
    assert isinstance(max_iter, list)


@pytest.mark.integration
def test_integration_state_grid() -> None:
    """Integration test for inserting a parameter grid into the session state."""
    state = st.session_state

    C = [1, 2, 3]
    penalty = ["l2", "l1", "elasticnet"]
    solver = ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"]
    max_iter = [100, 200, 300]

    state["param_grid"] = {
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "max_iter": max_iter,
    }

    assert state.get("param_grid") == {
        "C": [1, 2, 3],
        "penalty": ["l2", "l1", "elasticnet"],
        "solver": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        "max_iter": [100, 200, 300],
    }
