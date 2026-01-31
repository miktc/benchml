import json
import os
import pytest


@pytest.mark.integration
def test_integration_json_params() -> None:
    """Integration test to create and read a JSON file named `test_param_grid.json`."""
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [1, 3],
        "min_samples_split": [2, 4],
    }

    with open("test_param_grid.json", "w", newline="\n") as file:
        json.dump(grid, file, indent=4)

    assert os.path.exists("test_param_grid.json")

    with open("test_param_grid.json", "r", newline="\n") as file:
        data = json.load(file)

        assert list(data.keys()) == ["n_estimators", "max_depth", "min_samples_split"]
        assert data["n_estimators"] == [100, 200]
        assert data["max_depth"] == [1, 3]
        assert data["min_samples_split"] == [2, 4]


@pytest.mark.integration
def test_integration_rm_json() -> None:
    """Integration test to remove the JSON test file named `test_param_grid.json`."""
    if os.path.exists("test_data.json"):
        os.remove("test_data.json")

    assert not os.path.exists("test_data.json")

    if os.path.exists("test_param_grid.json"):
        os.remove("test_param_grid.json")

    assert not os.path.exists("test_param_grid.json")
