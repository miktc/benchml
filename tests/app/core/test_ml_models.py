from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def test_load_data() -> None:
    """Test the shapes of `X` and `y` after generating synthetic data using `make_classification`."""
    X, y = make_classification(n_samples=100, n_features=25, random_state=42)
    assert X.shape == (100, 25)
    assert y.shape == (100,)


def test_train_classification_model() -> None:
    """Test that attributes from the classification model `RandomForestClassifier` exist and are accessible
    after generating synthetic data using `make_classification` to train the model.
    """
    X, y = make_classification(n_samples=100, n_features=25, random_state=42)
    rfc = RandomForestClassifier(random_state=42).fit(X, y)
    assert hasattr(rfc, "n_estimators")
    assert hasattr(rfc, "feature_importances_")
    assert callable(getattr(rfc, "predict"))


def test_train_regression_model() -> None:
    """Test that attributes from the regression model `RandomForestRegressor` exist and are accessible
    after generating synthetic data using `make_classification` to train the model.
    """
    X, y, coef = make_regression(
        n_samples=100, n_features=25, random_state=42, coef=True
    )
    rfr = RandomForestRegressor(random_state=42).fit(X, y)
    assert hasattr(rfr, "estimator_")
    assert hasattr(rfr, "feature_importances_")
    assert callable(getattr(rfr, "predict"))
