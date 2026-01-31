import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import convert_type, parse_values
from typing import Any, cast


def mlpr_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `MLPRegressor` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    MLPR_hidden_layer_numbers = st.selectbox(
        label="Number of Hidden Layer Sizes:", options=["1", "2", "3"]
    )

    if MLPR_hidden_layer_numbers == "1":
        MLPR_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer:", value=f"{50},{50}", placeholder="Example: 50,50"
        )
        split_values = MLPR_hidden_layer_sizes1.split(",")

        if len(split_values) == 1:
            MLPR_hidden_layer_sizes_convert = convert_type(
                variable=MLPR_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list1: list[Any] = list(
                (MLPR_hidden_layer_sizes_convert,)
            )
        else:
            MLPR_hidden_layer_sizes_list1 = []
            for val in split_values:
                if val.strip() == "":
                    continue
                converted_val = convert_type(variable=val, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list1.append(converted_val)

        MLPR_hidden_layer_sizes_list = [tuple(MLPR_hidden_layer_sizes_list1)]
    elif MLPR_hidden_layer_numbers == "2":
        MLPR_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer 1:", value=f"{50}", placeholder="Example: 50"
        )
        split_values1 = MLPR_hidden_layer_sizes1.split(",")

        if len(split_values1) == 1:
            MLPR_hidden_layer_sizes1_convert = convert_type(
                variable=MLPR_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list1 = list((MLPR_hidden_layer_sizes1_convert,))
        else:
            MLPR_hidden_layer_sizes_list1 = []
            for val1 in split_values1:
                if val1.strip() == "":
                    continue
                converted_val1 = convert_type(variable=val1, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list1.append(converted_val1)

        MLPR_hidden_layer_sizes2 = st.text_input(
            label="Hidden Layer 2:",
            value=f"{64},{32}",
            placeholder="Example: 64,32",
        )
        split_values2 = MLPR_hidden_layer_sizes2.split(",")

        if len(split_values2) == 1:
            MLPR_hidden_layer_sizes2_convert = convert_type(
                variable=MLPR_hidden_layer_sizes2, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list2 = list((MLPR_hidden_layer_sizes2_convert,))
        else:
            MLPR_hidden_layer_sizes_list2 = []
            for val2 in split_values2:
                if val2.strip() == "":
                    continue
                converted_val2 = convert_type(variable=val2, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list2.append(converted_val2)

        MLPR_hidden_layer_sizes_list = [
            tuple(MLPR_hidden_layer_sizes_list1),
            tuple(MLPR_hidden_layer_sizes_list2),
        ]
    elif MLPR_hidden_layer_numbers == "3":
        MLPR_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer 1:", value=f"{50}", placeholder="Example: 50"
        )
        split_values1 = MLPR_hidden_layer_sizes1.split(",")

        if len(split_values1) == 1:
            MLPR_hidden_layer_sizes1_convert = convert_type(
                variable=MLPR_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list1 = list((MLPR_hidden_layer_sizes1_convert,))
        else:
            MLPR_hidden_layer_sizes_list1 = []
            for val1 in split_values1:
                if val1.strip() == "":
                    continue
                converted_val1 = convert_type(variable=val1, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list1.append(converted_val1)

        MLPR_hidden_layer_sizes2 = st.text_input(
            label="Hidden Layer 2:", value=f"{64},{32}", placeholder="Example: 64,32"
        )
        split_values2 = MLPR_hidden_layer_sizes2.split(",")

        if len(split_values2) == 1:
            MLPR_hidden_layer_sizes2_convert = convert_type(
                variable=MLPR_hidden_layer_sizes2, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list2 = list((MLPR_hidden_layer_sizes2_convert,))
        else:
            MLPR_hidden_layer_sizes_list2 = []
            for val2 in split_values2:
                if val2.strip() == "":
                    continue
                converted_val2 = convert_type(variable=val2, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list2.append(converted_val2)

        MLPR_hidden_layer_sizes3 = st.text_input(
            label="Hidden Layer 3:",
            value=f"{128},{64},{32}",
            placeholder="Example: 128,64,32",
        )
        split_values3 = MLPR_hidden_layer_sizes3.split(",")

        if len(split_values3) == 1:
            MLPR_hidden_layer_sizes3_convert = convert_type(
                variable=MLPR_hidden_layer_sizes3, dtype=int, value=1
            )
            MLPR_hidden_layer_sizes_list3 = list((MLPR_hidden_layer_sizes3_convert,))
        else:
            MLPR_hidden_layer_sizes_list3 = []
            for val3 in split_values3:
                if val3.strip() == "":
                    continue
                converted_val3 = convert_type(variable=val3, dtype=int, value=1)
                MLPR_hidden_layer_sizes_list3.append(converted_val3)

        MLPR_hidden_layer_sizes_list = [
            tuple(MLPR_hidden_layer_sizes_list1),
            tuple(MLPR_hidden_layer_sizes_list2),
            tuple(MLPR_hidden_layer_sizes_list3),
        ]

    MLPR_activation = st.text_input(
        label="Activation:",
        value="relu,identity,logistic,tanh",
        placeholder="Example: relu,identity,logistic,tanh",
    )
    MLPR_activation_list = parse_values(param=MLPR_activation)

    MLPR_solver = st.text_input(
        label="Solver:", value="adam,lbfgs,sgd", placeholder="Example: adam,lbfgs,sgd"
    )
    MLPR_solver_list = parse_values(param=MLPR_solver)

    MLPR_alpha = st.text_input(
        label="Alpha:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    MLPR_alpha_list = parse_values(
        param=MLPR_alpha, convert=True, param_type=float, min_val=0
    )

    MLPR_max_iter = st.text_input(
        label="Max Iter:", value="200,400,600", placeholder="Example: 200,400,600"
    )
    MLPR_max_iter_list = parse_values(
        param=MLPR_max_iter, convert=True, param_type=int, min_val=1
    )

    MLPR_grid = {
        "hidden_layer_sizes": MLPR_hidden_layer_sizes_list,
        "activation": MLPR_activation_list,
        "solver": MLPR_solver_list,
        "alpha": MLPR_alpha_list,
        "max_iter": MLPR_max_iter_list,
    }
    state["param_grid"] = MLPR_grid

    if state.get("display_json"):
        st.json(MLPR_grid, expanded=1)

    return MLPR_grid
