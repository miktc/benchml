import streamlit as st

from app.core.central import CENTRAL
from typing import Any, cast, Optional, Type


def convert_type(
    variable: str, dtype: Type[int] | Type[float], value: int | float
) -> str | int | float:
    """Converts a variable to the appropriate type and checks if it is greater than a specified minimum value.

    Args:
        variable (str): Name of the variable to convert.
        dtype (int | float): The type to convert the variable to.
        value (int | float): The minimum value that the variable must be greater than.

    Returns:
        (str | int | float): The variable with the appropriate type.
    """
    state = cast(CENTRAL, st.session_state)

    converted_variable = variable
    try:
        converted_variable = dtype(variable)  # type: ignore
        if not isinstance(converted_variable, str):
            if converted_variable < value:
                state["valid_entry"] = False
    except ValueError:
        state["valid_entry"] = False

    return converted_variable


def parse_values(
    param: str,
    convert: bool = False,
    param_type: Optional[Type[int] | Type[float]] = None,
    min_val: Optional[int | float] = None,
) -> list[Any]:
    """Splits a string of values into separate elements and appends each element into a list with its appropriate type.

    Args:
        param (str): A single string of values.
        convert (bool, optional): True to convert `param` into values. Defaults to False.
        param_type (Optional[int | float], optional): Type int or float. Defaults to None.
        min_val (Optional[int | float], optional): Minimum value of the `param` parameter. Defaults to None.

    Returns:
        list[Any]: List of elements where each element has the appropriate type.
    """
    str_elements = param.split(",")
    param_list = []
    for i in str_elements:
        if convert:
            if param_type is not None and min_val is not None:
                converted_param = convert_type(
                    variable=i, dtype=param_type, value=min_val
                )
                param_list.append(converted_param)
        else:
            param_list.append(i)

    return param_list
