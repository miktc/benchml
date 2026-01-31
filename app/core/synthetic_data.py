import numpy as np
import streamlit as st

from app.core.central import CENTRAL
from typing import cast


def add_feature_cols() -> None:
    """Adds synthetic features to the dataset.

    Notes:
        If the required session state key is None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    origin_df = state.get("df")
    add_feature_noise = state.get("add_feature_noise")

    if add_feature_noise is None:
        st.error(
            body="Session state key 'add_feature_noise' must not be None to add features!"
        )
        return

    if origin_df is not None:
        df = origin_df.copy()
        rng = np.random.default_rng(42)

        for i in range(add_feature_noise):
            df[f"noise_{i+1}"] = rng.integers(
                df.values.min(), df.values.max(), size=(len(df))
            )

        state["df"] = df


def add_rows() -> None:
    """Adds synthetic rows to the dataset.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    origin_df = state.get("df")
    advanced_settings = state.get("advanced_settings")
    add_row_noise = state.get("add_row_noise")
    select_target_value = state.get("select_target_value")
    target = state.get("target", "")
    target_value = state.get("target_value")

    if not advanced_settings:
        if add_row_noise is None or add_row_noise < 0:
            st.error(
                body="Session state key 'add_row_noise' must not be None or negative to add rows!"
            )
            return
    else:
        if (
            add_row_noise is None
            or add_row_noise < 0
            or select_target_value is None
            or target_value is None
        ):
            st.error(
                body="Session state keys must not be None or negative to add rows!"
            )
            return

    if origin_df is not None:
        df = origin_df.copy()

        for _ in range(add_row_noise):
            while True:
                if state.get("model_type_select") == "Classification":
                    create_random_nums = [
                        (
                            df[col].min()
                            if df[col].min() == df[col].max()
                            else np.random.randint(df[col].min(), df[col].max() + 1)
                        )
                        for col in df.columns
                    ]

                    if select_target_value == "Yes" and target in df:
                        create_random_nums[df.columns.get_loc(target)] = target_value

                    if not np.array_equal(
                        create_random_nums, np.array(df.iloc[-1].values)
                    ):
                        df.loc[len(df)] = create_random_nums
                        break
                else:
                    create_random_floats = [
                        (
                            df[col].min()
                            if df[col].min() == df[col].max()
                            else np.random.uniform(df[col].min(), df[col].max())
                        )
                        for col in df.columns
                    ]

                    if select_target_value == "Yes" and target in df:
                        create_random_floats[df.columns.get_loc(target)] = target_value

                    if not np.array_equal(
                        create_random_floats, np.array(df.iloc[-1].values)
                    ):
                        df.loc[len(df)] = create_random_floats
                        break

        state["df"] = df
