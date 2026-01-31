import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import streamlit as st

from app.core.central import CENTRAL
from sklearn.metrics import (
    auc,
    average_precision_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)
from typing import cast


def create_cm(display: bool = False, save: bool = False) -> None:
    """Creates the confusion matrix.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("confusion_matrix.png"):
            os.remove("confusion_matrix.png")

    y_test = state.get("y_test")
    y_pred = state.get("y_pred")

    if y_test is not None and y_pred is not None:
        fig, ax = plt.subplots()
        plt.title("Confusion Matrix")
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("confusion_matrix.png")


def create_roc_curve(display: bool = False, save: bool = False) -> None:
    """Creates the ROC curve.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("roc_curve.png"):
            os.remove("roc_curve.png")

    y_test = state.get("y_test")
    y_pred_prob = state.get("y_pred_prob")

    if y_test is not None and y_pred_prob is not None:
        fig, ax = plt.subplots()
        class_count = y_pred_prob.shape[1] if len(y_pred_prob.shape) > 1 else 2
        for i in range(class_count):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {i} (AUC = {round(roc_auc,4)})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(
            xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve"
        )
        ax.legend()

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("roc_curve.png")


def create_pr_curve(display: bool = False, save: bool = False) -> None:
    """Creates the precision-recall curve.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("pr_curve.png"):
            os.remove("pr_curve.png")

    y_test = state.get("y_test")
    y_pred_prob = state.get("y_pred_prob")

    if y_test is not None and y_pred_prob is not None:
        fig, ax = plt.subplots()
        class_count = y_pred_prob.shape[1] if len(y_pred_prob.shape) > 1 else 2
        for i in range(class_count):
            precision, recall, _ = precision_recall_curve(
                y_test == i, y_pred_prob[:, i]
            )
            average_precision = average_precision_score(y_test == i, y_pred_prob[:, i])
            ax.plot(
                recall,
                precision,
                label=f"Class {i} = (AVG = {round(average_precision,4)})",
            )
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
        ax.legend()

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("pr_curve.png")


def create_shap(display: bool = False, save: bool = False) -> None:
    """Creates the SHAP summary plot.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("shap_summary.png"):
            os.remove("shap_summary.png")

    SHAP_values = state.get("SHAP_values")
    X_test = state.get("X_test")

    if SHAP_values is not None and X_test is not None:
        plt.title("SHAP Summary Plot")
        shap.summary_plot(
            SHAP_values, X_test, show=False, rng=np.random.default_rng(seed=42)
        )

    if display:
        st.pyplot(plt.gcf())

    if not display and save:
        plt.savefig("shap_summary.png")


def create_predict_actual(display: bool = False, save: bool = False) -> None:
    """Creates the predicted vs. actual values plot.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("predict_actual.png"):
            os.remove("predict_actual.png")

    y_test = state.get("y_test")
    y_pred = state.get("y_pred")

    if y_test is not None and y_pred is not None:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
        ax.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            linestyle="--",
            color="black",
            label="Ideal",
        )
        ax.set(
            xlabel="Actual Values",
            ylabel="Predicted Values",
            title="Predicted vs. Actual Values",
        )
        ax.legend()

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("predict_actual.png")


def create_error_distribution(display: bool = False, save: bool = False) -> None:
    """Creates the error distribution plot.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("error_distribution.png"):
            os.remove("error_distribution.png")

    y_test = state.get("y_test")
    y_pred = state.get("y_pred")

    if y_test is not None and y_pred is not None:
        error = y_test - y_pred
        fig, ax = plt.subplots()
        ax.hist(error, bins=20, color="blue", alpha=0.6)
        ax.axvline(0, color="red", linestyle="--", label="Zero Error")
        ax.set(
            xlabel="Prediction Error",
            ylabel="Frequency",
            title="Error Distribution Plot",
        )
        ax.legend()

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("error_distribution.png")


def create_residual(display: bool = False, save: bool = False) -> None:
    """Creates the residual plot.

    Args:
        display (bool, optional): True to display the visualization. Defaults to False.
        save (bool, optional): True to save the visualization as a PNG file. Defaults to False.
    """
    state = cast(CENTRAL, st.session_state)

    if not display:
        if os.path.exists("residual.png"):
            os.remove("residual.png")

    y_test = state.get("y_test")
    y_pred = state.get("y_pred")

    if y_test is not None and y_pred is not None:
        residual = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residual, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--", label="Zero Residual")
        ax.set(xlabel="Predicted Values", ylabel="Residuals", title="Residual Plot")
        ax.legend()

    if display:
        st.pyplot(fig)

    if not display and save:
        fig.savefig("residual.png")
