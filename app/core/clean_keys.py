import streamlit as st


def clean_keys() -> None:
    """Removes keys from the Streamlit session state."""
    remove_keys = [
        "X_test",
        "y_test",
        "y_pred",
        "y_pred_prob",
        "SHAP_values",
        "predict_form",
        "predict_result",
        "param_grid",
        "all_metrics",
        "scale",
        "confidence",
        "uncertainty",
        "best_params",
        "search_method",
        "select_cv",
        "select_random_state",
        "select_target_value",
        "uploaded_file_name",
        "sql_toggle",
        "config_duration",
        "time_created",
        "row_noise_percent",
        "feature_noise_percent",
        "target_value",
    ]

    for key in remove_keys:
        if key in st.session_state:
            del st.session_state[key]
