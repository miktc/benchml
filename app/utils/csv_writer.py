import csv
import streamlit as st


def CSV_Writer() -> None:
    """Creates a CSV file for the dataset named `dataset.csv`.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    df = st.session_state.get("df")
    if df is None:
        st.error(body="No data is available to write a CSV file!")
        return

    with open("dataset.csv", "w", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        writer.writerows(df.values)
