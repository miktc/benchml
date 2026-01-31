from app.utils.convert import convert_type, parse_values


def test_convert_type() -> None:
    """Test that the `convert_type` function converts variables to the correct data type."""
    variable1 = "1"
    variable2 = "2.5"
    variable3 = "string"
    variable4 = "string2"

    assert convert_type(variable=variable1, dtype=int, value=0) == 1
    assert convert_type(variable=variable2, dtype=float, value=0) == 2.5
    assert convert_type(variable=variable3, dtype=int, value=0) == "string"
    assert convert_type(variable=variable4, dtype=int, value=0) == "string2"


def test_parse_values() -> None:
    """Test that the `parse_values` function parses a string of values to create a list
    where each element in the list has the correct data type.
    """
    values1 = "1,2,3"
    values2 = "1.5,2.5,3.5"
    values3 = "1,2.5,3"
    values4 = "test,test2"
    values5 = "1,2,test"

    assert parse_values(param=values1, convert=True, param_type=int, min_val=0) == [
        1,
        2,
        3,
    ]
    assert parse_values(param=values2, convert=True, param_type=float, min_val=0) == [
        1.5,
        2.5,
        3.5,
    ]
    assert parse_values(param=values3, convert=True, param_type=float, min_val=0) == [
        1,
        2.5,
        3,
    ]
    assert parse_values(param=values4, convert=False) == ["test", "test2"]
    assert parse_values(param=values5, convert=True, param_type=int, min_val=0) == [
        1,
        2,
        "test",
    ]
