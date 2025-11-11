from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from src.data import get_class_labels
from src.utils import make_dir


def evaluate_classical_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    label_encoder: LabelEncoder = None,
) -> dict:
    """Evaluate classical ML model (Random Forest or XGBoost).

    Args:
        X: Data samples
        y_true: Ground-truth labels
        label_encoder: LabelEncoder object to map labels (optional)

    Returns:
        Dict containing evaluation metrics
    """
    y_pred = model.predict(X)
    accuracy = model.score(X, y_true)

    if label_encoder is not None:
        y_true_decoded = label_encoder.inverse_transform(y_true)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
    else:
        y_true_decoded = y_true
        y_pred_decoded = y_pred

    cm = confusion_matrix(y_true_decoded, y_pred_decoded)

    report = classification_report(
        y_true_decoded, y_pred_decoded, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "predictions": y_pred,
        "y_true": y_true_decoded,
        "y_pred": y_pred_decoded,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def save_evaluation_metrics(
    results: dict,
    output_dir: Path,
    split_name: str = "test",
    model_name: str = "model",
    group_activities: bool = False,
) -> None:
    """Save evaluation metrics and figures.

    Args:
        results: Dict containing evaluation metrics
        output_dir: Directory to save results
        split_name: Name of data split evaluated
        model_name: Name of model used for evaluation
    """
    output_dir = make_dir(output_dir)

    cm_path = output_dir / f"cm_{split_name}.npy"
    np.save(cm_path, results["confusion_matrix"])

    pred_path = output_dir / f"predictions_{split_name}.npz"
    np.savez_compressed(
        pred_path,
        y_true=results["y_true"],
        y_pred=results["y_pred"],
    )
    print(f"Predictions saved to: {pred_path}")

    unique_ids = sorted(
        np.unique(np.concatenate([results["y_true"], results["y_pred"]]))
    )
    display_labels = get_class_labels(unique_ids, intensity_groups=group_activities)

    plot_path = output_dir / f"cm_{split_name}.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"], display_labels=display_labels
    )

    cmap = "Blues" if "forest" in model_name.lower() else "Greens"
    disp.plot(
        ax=ax, cmap=cmap, colorbar=True, text_kw={"fontsize": 10, "fontweight": "bold"}
    )

    title = f"{model_name} - {split_name.capitalize()} Set"
    if "accuracy" in results:
        title += f"\nAccuracy: {results['accuracy']:.2%}"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix plot saved to: {plot_path}")
