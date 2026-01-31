import numpy as np
import pandas as pd
import shap
import streamlit as st

from app.core.central import CENTRAL
from app.utils.csv_writer import CSV_Writer
from app.utils.json_writer import JSON_Writer
from scipy.special import expit
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample
from typing import Any, cast, Optional


def models(mode: Optional[str] = None) -> None:
    """Trains a machine learning model and calculates metrics, SHAP values, and predictions based on a user's selection.

    Args:
        mode (str, optional): Setting the mode to `"predict"` will call the internal predict function after
            performing training. Defaults to None.

    Notes:
        If the required session state keys are None, a Streamlit error message is displayed using `st.error`
        and the function returns early.
    """
    state = cast(CENTRAL, st.session_state)

    model_type_select = state.get("model_type_select")
    select_model = state.get("select_model")
    advanced_setting = state.get("advanced_setting")
    search_method = state.get("search_method")
    param_grid = state.get("param_grid")
    cv = state.get("select_cv")
    random_state = state.get("select_random_state", 42)
    state["scale"] = 0

    models_dict: dict[str, tuple[BaseEstimator, int]] = {
        "LogisticRegression": (LogisticRegression(), 1),
        "DecisionTreeClassifier": (DecisionTreeClassifier(), 0),
        "LinearSVC": (LinearSVC(), 1),
        "RandomForestClassifier": (RandomForestClassifier(), 0),
        "GradientBoostingClassifier": (GradientBoostingClassifier(), 0),
        "MLPClassifier": (MLPClassifier(), 1),
        "KNeighborsClassifier": (KNeighborsClassifier(), 1),
        "LinearRegression": (LinearRegression(), 1),
        "Ridge": (Ridge(), 1),
        "DecisionTreeRegressor": (DecisionTreeRegressor(), 0),
        "SVR": (SVR(), 1),
        "RandomForestRegressor": (RandomForestRegressor(), 0),
        "GradientBoostingRegressor": (GradientBoostingRegressor(), 0),
        "MLPRegressor": (MLPRegressor(), 1),
        "KNeighborsRegressor": (KNeighborsRegressor(), 1),
    }

    if model_type_select is None or select_model is None:
        st.error(
            body="Key 'model_type_select' or 'select_model' must not be None to get the model!"
        )
        return

    estimator, is_scaled = models_dict.get(select_model)  # type: ignore

    if (
        advanced_setting
        and search_method is not None
        and param_grid is not None
        and cv is not None
    ):
        model_with_search = (
            GridSearchCV(
                estimator, param_grid=param_grid, cv=cv, error_score=float("nan")
            )
            if search_method == "GridSearchCV"
            else RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                cv=cv,
                random_state=random_state,
                error_score=float("nan"),
            )
        )
    else:
        model_without_search = estimator

    model = model_with_search if advanced_setting else model_without_search
    model = cast(Any, model)

    state["scale"] = is_scaled

    # Write CSV before removing target column
    CSV_Writer()
    if advanced_setting:
        JSON_Writer()

    X = state.get("df")
    target = state.get("target")

    if X is None or target is None:
        st.error(body="DataFrame 'df' or target selection must not be None!")
        return

    y = X.pop(target)
    st.session_state["X_features"] = X

    test_size_val = state.get("test_size")

    if test_size_val is None:
        st.error(body="Test size value must not be None!")
        return

    test_size = (test_size_val) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if state["scale"] == 1:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    state["X_test"] = X_test
    state["y_test"] = y_test

    # Using search methods
    if hasattr(model, "best_params_") and hasattr(model, "best_estimator_"):
        best_params = model.best_params_
        state["best_params"] = best_params
        model = model.best_estimator_

    def metrics() -> None:
        """Calculates metrics after model training based on the model type selected."""
        y_pred = model.predict(X_test)

        if model_type_select == "Classification":
            report = cast(dict, classification_report(y_test, y_pred, output_dict=True))
        else:
            report = {
                "r2_score": round(r2_score(y_test, y_pred), 2),
                "root_mean_squared_error": round(
                    root_mean_squared_error(y_test, y_pred), 2
                ),
                "mean_absolute_error": round(mean_absolute_error(y_test, y_pred), 2),
                "mean_squared_error": round(mean_squared_error(y_test, y_pred), 2),
            }

        metrics = report
        state["all_metrics"] = metrics

    def explain() -> None:
        """Calculates SHAP values after model training based on the model type selected."""
        if model_type_select == "Classification":
            if (
                select_model == "MLPClassifier"
                or select_model == "KNeighborsClassifier"
            ):
                explainer = shap.KernelExplainer(
                    model.predict_proba, shap.sample(X_train, 20)
                )
                shap_values = explainer.shap_values(X_test)
            elif select_model == "GradientBoostingClassifier":
                explainer = shap.Explainer(model.predict_proba, X_train)
                shap_values = explainer(X_test)
            else:
                explainer = shap.Explainer(model, X_test)
                if select_model == "RandomForestClassifier":
                    shap_values = explainer(X_test, check_additivity=False)
                else:
                    shap_values = explainer(X_test)

            state["SHAP_values"] = shap_values

            y_pred = model.predict(X_test)
            state["y_pred"] = y_pred

            # Check attribute for LinearSVC
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X_test)
                state["y_pred_prob"] = y_pred_prob
        else:
            if (
                select_model == "MLPRegressor"
                or select_model == "KNeighborsRegressor"
                or select_model == "SVR"
            ):
                explainer = shap.KernelExplainer(
                    model.predict, shap.sample(X_train, 20)
                )
                shap_values = explainer.shap_values(X_test)
            elif (
                select_model == "DecisionTreeRegressor"
                or select_model == "RandomForestRegressor"
                or select_model == "GradientBoostingRegressor"
            ):
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test, check_additivity=False)
            else:
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)

            state["SHAP_values"] = shap_values

            y_pred = model.predict(X_test)
            state["y_pred"] = y_pred

    def predict() -> None:
        """Generates predictions based on user input after model training.

        Notes:
            If required session state keys are None, a Streamlit error message is displayed using `st.error`
            and the function returns early.
        """
        user_input_form = state.get("predict_form")
        df = state.get("df")

        if df is None:
            st.error(body="DataFrame 'df' must not be None for prediction!")
            return

        # Scale Prediction
        if state.get("predict_form") and state["scale"] == 1:
            scale_live_predict = scaler.transform(pd.DataFrame(user_input_form))
            pred_df = pd.DataFrame(
                scale_live_predict, columns=[col for col in df.columns]
            )
            predicted_result = model.predict(pred_df)
            state["predict_result"] = predicted_result[0]
        elif state.get("predict_form") and state["scale"] != 1:
            pred_df = pd.DataFrame(user_input_form)
            predicted_result = model.predict(pred_df)
            state["predict_result"] = predicted_result[0]

        if model_type_select == "Classification":
            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(pred_df)
                state["confidence"] = f"{round((np.max(confidence,axis=1)[0]*100),2)}%"
            elif hasattr(model, "decision_function"):
                confidence = model.decision_function(pred_df)
                probs = expit(confidence)
                predicted_result = model.predict(pred_df)[0]
                if probs.ndim == 1:
                    state["confidence"] = (
                        f"{round(probs[0] * 100, 2)}%"
                        if predicted_result == 1
                        else f"{round((1 - probs[0]) * 100, 2)}%"
                    )
                else:
                    index_value = list(model.classes_).index(predicted_result)
                    state["confidence"] = f"{round(probs[0, index_value] * 100, 2)}%"
            else:
                confidence = None
        else:
            preds_list = []

            if X_train is None or y_train is None:
                st.error(body="X_train or y_train must not be None during resampling!")
                return

            for _ in range(101):
                X_resample, y_resample = resample(X_train, y_train)  # type: ignore
                clone_model = clone(model)
                clone_model.fit(X_resample, y_resample)
                preds_list.append(clone_model.predict(pred_df))
            preds = np.array(preds_list)
            y_pred = preds.mean(axis=0)[0]
            y_pred_rounded = round(y_pred, 4)
            error = preds.std(axis=0)[0]
            state["predict_result"] = y_pred_rounded
            state["uncertainty"] = f"±{error:.04f}"

    if mode == "predict":
        predict()
    else:
        metrics()
        explain()
