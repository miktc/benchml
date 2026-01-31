import streamlit as st

from app.core.central import CENTRAL
from app.utils.convert import convert_type, parse_values
from typing import Any, cast


def mlpc_grid() -> dict[str, list[Any]]:
    """Renders the parameter grid UI for a `MLPClassifier` model.

    Returns:
        dict[str, list[Any]]
            A parameter grid as a dictionary where the key is a string
            and the value is a list containing any type.
    """
    state = cast(CENTRAL, st.session_state)

    MLPC_hidden_layer_numbers = st.selectbox(
        label="Number of Hidden Layer Sizes:", options=["1", "2", "3"]
    )

    if MLPC_hidden_layer_numbers == "1":
        MLPC_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer:", value=f"{50},{50}", placeholder="Example: 50,50"
        )
        split_values = MLPC_hidden_layer_sizes1.split(",")

        if len(split_values) == 1:
            MLPC_hidden_layer_sizes_convert = convert_type(
                variable=MLPC_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list1 = list((MLPC_hidden_layer_sizes_convert,))
        else:
            MLPC_hidden_layer_sizes_list1 = []
            for val in split_values:
                if val.strip() == "":
                    continue
                converted_val = convert_type(variable=val, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list1.append(converted_val)

        MLPC_hidden_layer_sizes_list = [tuple(MLPC_hidden_layer_sizes_list1)]
    elif MLPC_hidden_layer_numbers == "2":
        MLPC_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer 1:", value=f"{50}", placeholder="Example: 50"
        )
        split_values1 = MLPC_hidden_layer_sizes1.split(",")

        if len(split_values1) == 1:
            MLPC_hidden_layer_sizes1_convert = convert_type(
                variable=MLPC_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list1 = list((MLPC_hidden_layer_sizes1_convert,))
        else:
            MLPC_hidden_layer_sizes_list1 = []
            for val1 in split_values1:
                if val1.strip() == "":
                    continue
                converted_val1 = convert_type(variable=val1, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list1.append(converted_val1)

        MLPC_hidden_layer_sizes2 = st.text_input(
            label="Hidden Layer 2:",
            value=f"{64},{32}",
            placeholder="Example: 64,32",
        )
        split_values2 = MLPC_hidden_layer_sizes2.split(",")

        if len(split_values2) == 1:
            MLPC_hidden_layer_sizes2_convert = convert_type(
                variable=MLPC_hidden_layer_sizes2, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list2 = list((MLPC_hidden_layer_sizes2_convert,))
        else:
            MLPC_hidden_layer_sizes_list2 = []
            for val2 in split_values2:
                if val2.strip() == "":
                    continue
                converted_val2 = convert_type(variable=val2, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list2.append(converted_val2)

        MLPC_hidden_layer_sizes_list = [
            tuple(MLPC_hidden_layer_sizes_list1),
            tuple(MLPC_hidden_layer_sizes_list2),
        ]
    elif MLPC_hidden_layer_numbers == "3":
        MLPC_hidden_layer_sizes1 = st.text_input(
            label="Hidden Layer 1:", value=f"{50}", placeholder="Example: 50"
        )
        split_values1 = MLPC_hidden_layer_sizes1.split(",")

        if len(split_values1) == 1:
            MLPC_hidden_layer_sizes1_convert = convert_type(
                variable=MLPC_hidden_layer_sizes1, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list1 = list((MLPC_hidden_layer_sizes1_convert,))
        else:
            MLPC_hidden_layer_sizes_list1 = []
            for val1 in split_values1:
                if val1.strip() == "":
                    continue
                converted_val1 = convert_type(variable=val1, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list1.append(converted_val1)

        MLPC_hidden_layer_sizes2 = st.text_input(
            label="Hidden Layer 2:", value=f"{64},{32}", placeholder="Example: 64,32"
        )
        split_values2 = MLPC_hidden_layer_sizes2.split(",")

        if len(split_values2) == 1:
            MLPC_hidden_layer_sizes2_convert = convert_type(
                variable=MLPC_hidden_layer_sizes2, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list2 = list((MLPC_hidden_layer_sizes2_convert,))
        else:
            MLPC_hidden_layer_sizes_list2 = []
            for val2 in split_values2:
                if val2.strip() == "":
                    continue
                converted_val2 = convert_type(variable=val2, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list2.append(converted_val2)

        MLPC_hidden_layer_sizes3 = st.text_input(
            label="Hidden Layer 3:",
            value=f"{128},{64},{32}",
            placeholder="Example: 128,64,32",
        )
        split_values3 = MLPC_hidden_layer_sizes3.split(",")

        if len(split_values3) == 1:
            MLPC_hidden_layer_sizes3_convert = convert_type(
                variable=MLPC_hidden_layer_sizes3, dtype=int, value=1
            )
            MLPC_hidden_layer_sizes_list3 = list((MLPC_hidden_layer_sizes3_convert,))
        else:
            MLPC_hidden_layer_sizes_list3 = []
            for val3 in split_values3:
                if val3.strip() == "":
                    continue
                converted_val3 = convert_type(variable=val3, dtype=int, value=1)
                MLPC_hidden_layer_sizes_list3.append(converted_val3)

        MLPC_hidden_layer_sizes_list = [
            tuple(MLPC_hidden_layer_sizes_list1),
            tuple(MLPC_hidden_layer_sizes_list2),
            tuple(MLPC_hidden_layer_sizes_list3),
        ]

    MLPC_activation = st.text_input(
        label="Activation:",
        value="relu,identity,logistic,tanh",
        placeholder="Example: relu,identity,logistic,tanh",
    )
    MLPC_activation_list = parse_values(param=MLPC_activation)

    MLPC_solver = st.text_input(
        label="Solver:", value="adam,lbfgs,sgd", placeholder="Example: adam,lbfgs,sgd"
    )
    MLPC_solver_list = parse_values(param=MLPC_solver)

    MLPC_alpha = st.text_input(
        label="Alpha:", value="0.001,0.01,0.1", placeholder="Example: 0.001,0.01,0.1"
    )
    MLPC_alpha_list = parse_values(
        param=MLPC_alpha, convert=True, param_type=float, min_val=0
    )

    MLPC_max_iter = st.text_input(
        label="Max Iter:", value="200,400,600", placeholder="Example: 200,400,600"
    )
    MLPC_max_iter_list = parse_values(
        param=MLPC_max_iter, convert=True, param_type=int, min_val=1
    )

    MLPC_grid = {
        "hidden_layer_sizes": MLPC_hidden_layer_sizes_list,
        "activation": MLPC_activation_list,
        "solver": MLPC_solver_list,
        "alpha": MLPC_alpha_list,
        "max_iter": MLPC_max_iter_list,
    }
    state["param_grid"] = MLPC_grid

    if state.get("display_json"):
        st.json(MLPC_grid, expanded=1)

    return MLPC_grid
