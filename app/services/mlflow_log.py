import app.core.visuals as visuals
import mlflow
import os
import requests
import streamlit as st

from app.core.central import CENTRAL
from app.utils.try_util import try_except
from dotenv import load_dotenv
from typing import cast


def run_mlflow(test: bool = False) -> bool:
    """Tests the connection to MLflow and logs experiments when the connection is successful.

    Args:
        test (bool, optional): True to call the internal `connection` function
            without logging experiments to MLflow. Defaults to False.

    Returns:
        bool: True if the connection to MLflow is successful, False otherwise.
    """
    state = cast(CENTRAL, st.session_state)

    load_dotenv()
    mlflow_uri = os.getenv("MLFLOWURI") or ""

    def connection() -> bool:
        """Tests the connection to MLflow.

        Returns:
            bool: True if the connection to MLflow is successful, False otherwise.
        """
        try:
            response = requests.get(mlflow_uri)
            if response.status_code == 200:
                result = True
            else:
                state["mlflow_toggle"] = False
        except requests.exceptions.ConnectionError:
            state["mlflow_toggle"] = False
            result = False

        return result

    def log_mlflow() -> None:
        """Logs the experiment to MLflow.

        Notes:
            If the required session state keys are None, a Streamlit error message is displayed using `st.error`
            and the function returns early.
        """
        select_model = state.get("select_model")
        advanced_setting = state.get("advanced_setting")
        metrics = state.get("all_metrics")
        search_method = state.get("search_method")
        best_params = state.get("best_params")

        if not advanced_setting:
            if select_model is None or metrics is None:
                st.error(
                    body="Session state keys must not be None before MLflow logging!"
                )
                return
        else:
            if (
                best_params is None
                or select_model is None
                or search_method is None
                or metrics is None
            ):
                st.error(
                    body="Session state keys must not be None before MLflow logging!"
                )
                return

        advanced_setting = cast(bool, advanced_setting)
        best_params = cast(dict, best_params)
        select_model = cast(str, select_model)
        search_method = cast(str, search_method)
        metrics = cast(dict, metrics)

        if not advanced_setting:
            desc_msg = f"The experiment was created with a {select_model} model."
        else:
            desc_msg = f"The experiment was created with a {select_model} model using {search_method} as the search method."

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name="BenchML")

        with mlflow.start_run(description=desc_msg):
            if state.get("model_type_select") == "Classification":
                try_except(
                    lambda: visuals.create_shap(display=False, save=True),
                    lambda: mlflow.log_artifact("shap_summary.png"),
                )
                try_except(
                    lambda: visuals.create_cm(display=False, save=True),
                    lambda: mlflow.log_artifact("confusion_matrix.png"),
                )
                try_except(
                    lambda: visuals.create_roc_curve(display=False, save=True),
                    lambda: mlflow.log_artifact("roc_curve.png"),
                )
                try_except(
                    lambda: visuals.create_pr_curve(display=False, save=True),
                    lambda: mlflow.log_artifact("pr_curve.png"),
                )

                for k, v in metrics.items():
                    if hasattr(v, "items") is False:
                        mlflow.log_metric(k, round(v, 4))
                    try:
                        mlflow.log_metric(f"{k}_precision", round(v["precision"], 4))
                        mlflow.log_metric(f"{k}_recall", round(v["recall"], 4))
                        mlflow.log_metric(f"{k}_f1-score", round(v["f1-score"], 4))
                    except TypeError:
                        pass
            else:
                try_except(
                    lambda: visuals.create_shap(display=False, save=True),
                    lambda: mlflow.log_artifact("shap_summary.png"),
                )
                try_except(
                    lambda: visuals.create_predict_actual(display=False, save=True),
                    lambda: mlflow.log_artifact("predict_actual.png"),
                )
                try_except(
                    lambda: visuals.create_error_distribution(display=False, save=True),
                    lambda: mlflow.log_artifact("error_distribution.png"),
                )
                try_except(
                    lambda: visuals.create_residual(display=False, save=True),
                    lambda: mlflow.log_artifact("residual.png"),
                )
                mlflow.log_metrics(metrics)

            if advanced_setting:
                mlflow.log_params(best_params)
                mlflow.log_artifact("param_grid.json")

            mlflow.log_artifact("dataset.csv")

    if test:
        connection()
    else:
        if connection():
            log_mlflow()

    return connection()
