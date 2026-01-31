import csv
import os
import pytest


@pytest.mark.integration
def test_integration_csv_data() -> None:
    """Integration test to create and read a CSV test file named `test_data.csv`."""
    with open("test_data.csv", "w", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(["col1", "col2", "col3"])
        writer.writerow([1, 2, 3])

    assert os.path.exists("test_data.csv")

    with open("test_data.csv", "r") as file:
        read = file.readlines()
        assert read == ["col1,col2,col3\n", "1,2,3\n"]


@pytest.mark.integration
def test_integration_rm_csv() -> None:
    """Integration test to remove the CSV test file named `test_data.csv`."""
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")

    assert not os.path.exists("test_data.csv")
