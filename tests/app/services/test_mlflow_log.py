import mlflow
import pytest


@pytest.mark.integration
def test_integration_mlflow_metrics(tmp_path) -> None:
    """Integration test for checking logged metrics in MLflow.

    Args:
        tmp_path (Path): Temporary path for MLflow tracking URI.
    """
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("test_experiment")
    with mlflow.start_run():
        # First log
        mlflow.log_metric("F1-score", 0.90)
        mlflow.log_metric("Precision", 0.91)
        mlflow.log_metric("Recall", 0.92)
        # Second log
        mlflow.log_metric("F1-score", 0.80)
        mlflow.log_metric("Precision", 0.81)
        mlflow.log_metric("Recall", 0.82)
        run = mlflow.active_run()
        if run is not None:
            run_id = run.info.run_id

    client = mlflow.client.MlflowClient()

    f1_score = client.get_metric_history(run_id=run_id, key="F1-score")
    assert f1_score[0].value == 0.9

    precision = client.get_metric_history(run_id=run_id, key="Precision")
    assert precision[0].value == 0.91

    recall = client.get_metric_history(run_id=run_id, key="Recall")
    assert recall[0].value == 0.92

    f1_score2 = client.get_metric_history(run_id=run_id, key="F1-score")
    assert f1_score2[-1].value == 0.8

    precision2 = client.get_metric_history(run_id=run_id, key="Precision")
    assert precision2[-1].value == 0.81

    recall2 = client.get_metric_history(run_id=run_id, key="Recall")
    assert recall2[-1].value == 0.82


@pytest.mark.integration
def test_integration_mlflow_params(tmp_path) -> None:
    """Integration test for checking logged parameters in MLflow.

    Args:
        tmp_path (Path): Temporary path for MLflow tracking URI.
    """
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("test_experiment")
    with mlflow.start_run():
        mlflow.log_param("C", 1)
        mlflow.log_param("max_iter", 100)
        run = mlflow.active_run()
        if run is not None:
            run_id = run.info.run_id

    client = mlflow.client.MlflowClient()

    params = client.get_run(run_id=run_id)
    assert params.data.params == {"C": "1", "max_iter": "100"}
