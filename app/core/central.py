import numpy as np
import pandas as pd
import shap

from numpy.typing import NDArray
from typing import Any, TypedDict


class CENTRAL(TypedDict, total=False):
    """Centralized storage for variables used within the application.

    Attributes:
        df (pd.DataFrame): Dataframe used for adding rows and features.
        advanced_setting (bool): Indicates whether the advanced setting toggle is enabled.
        model_type_select (str): Type of machine learning model selected.
        select_model (str): Machine learning model selection.
        add_row_noise (int): Number of synthetic rows added to the dataset.
        add_feature_noise (int): Number of synthetic features added to the dataset.
        samples (int): Number of original samples in the dataset.
        test_size (int): Test size for `train_test_split` as an integer.
        select_random_state (int): Random state integer for the search method `RandomSearchCV`.
        select_cv (int): The CV integer for a search method.
        target (str): Name of the target variable.
        target_value (int | float | None): Value of the target variable when rows are added to the dataset.
        search_method (str): Search method selection.
        param_grid (dict[str, Any]): Parameter grid for a search method.
        scale (int): Indicates whether to scale the data before model training.
        all_metrics (dict[str, Any]): Output metrics after the model training.
        X_features (pd.DataFrame): Dataframe containing data from all features.
        predict_result (np.integer): Result from a model's prediction.
        original_config (dict[str, Any]): Snapshot of the current session state.
        update_config (dict[str, Any]): Snapshot of the updated session state after modification.
        SHAP_values (shap.Explanation): SHAP values for the model prediction.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series): Test target values.
        y_pred (NDArray[Any]): Predicted target values.
        y_pred_prob (NDArray[Any]): Predicted probabilities for classification models.
        confidence (str | None): Confidence score for classification models.
        uncertainty (str): Uncertainty score for regression models.
        best_params (dict[str, Any]): Best parameters after using a search method.
        select_sample_dataset (str): Name of the selected sample dataset.
        uploaded_file (bool): Indicates whether a file was uploaded.
        uploaded_file_name (str): Name of the uploaded file.
        render_after_upload (bool): Indicates whether to render a tab after a file upload.
        sample_data (NDArray[Any]): Sample dataset encompassing features and target data.
        sample_feature_names (list[str]): Names of features within the sample dataset.
        sample_target (NDArray[Any]): Target values from the sample dataset.
        display_json (bool): Indicates whether to display the parameter grid in JSON format.
        configure (bool): Indicates whether the model training is successful.
        valid_entry (bool): Indicates whether the parameter grid entries are valid.
        mlflow_toggle (bool): Indicates whether the MLflow toggle is enabled.
        predict_form (dict[str, list[int | float]]): Form for allowing user input of feature values for prediction.
        predict_button (bool): Indicates whether the predict button has been enabled.
        sql_toggle (bool): Indicates whether the SQL Storage toggle is enabled.
        config_duration (float): Duration in seconds of the configuration within the application.
        time_created (str): Date and time when the configuration is completed.
        row_noise_percent (int | float): Percentage of noise within the rows.
        feature_noise_percent (int | float): Percentage of noise within the features.
        select_target_value (str): Indicates whether to set the target values when adding rows.
        api_key (str): API key used for retrieving OpenAI responses.
    """

    df: pd.DataFrame
    advanced_setting: bool
    model_type_select: str
    select_model: str
    add_row_noise: int
    add_feature_noise: int
    samples: int
    test_size: int
    select_random_state: int
    select_cv: int
    target: str
    target_value: int | float | None
    search_method: str
    param_grid: dict[str, Any]
    scale: int
    all_metrics: dict[str, Any]
    X_features: pd.DataFrame
    predict_result: np.integer
    original_config: dict[str, Any]
    update_config: dict[str, Any]
    SHAP_values: shap.Explanation
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: NDArray[Any]
    y_pred_prob: NDArray[Any]
    confidence: str | None
    uncertainty: str
    best_params: dict[str, Any]
    select_sample_dataset: str
    uploaded_file: bool
    uploaded_file_name: str
    render_after_upload: bool
    sample_data: NDArray[Any]
    sample_feature_names: list[str]
    sample_target: NDArray[Any]
    display_json: bool
    configure: bool
    valid_entry: bool
    mlflow_toggle: bool
    predict_form: dict[str, list[int | float]]
    predict_button: bool
    sql_toggle: bool
    config_duration: float
    time_created: str
    row_noise_percent: int | float
    feature_noise_percent: int | float
    select_target_value: str
    api_key: str
