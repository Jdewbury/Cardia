import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import get_class_labels


def analyze_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    intensity_groups: bool = False,
    threshold: float = 0.6,
) -> dict:
    """Analyze prediction confidence and save results."""

    confidence = probabilities.max(axis=1)
    correct = y_pred == y_true

    confidence_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": confidence,
            "correct": correct,
        }
    )

    confidence_df.to_csv(output_dir / "confidence_analysis.csv", index=False)

    summary = _compute_summary_stats(confidence_df, threshold)
    _print_confidence_summary(summary, confidence_df, y_true, intensity_groups)
    _plot_confidence_analysis(confidence_df, threshold, output_dir)

    with open(output_dir / "confidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _compute_summary_stats(df: pd.DataFrame, threshold: float) -> dict:
    """Compute confidence summary statistics."""
    low_conf = df[df["confidence"] < threshold]
    high_conf = df[df["confidence"] >= threshold]

    return {
        "overall_accuracy": float(df["correct"].mean()),
        "mean_confidence_correct": float(df[df["correct"]]["confidence"].mean()),
        "mean_confidence_incorrect": float(df[~df["correct"]]["confidence"].mean()),
        "threshold": threshold,
        "low_confidence_count": int(len(low_conf)),
        "low_confidence_accuracy": (
            float(low_conf["correct"].mean()) if len(low_conf) > 0 else 0
        ),
        "high_confidence_count": int(len(high_conf)),
        "high_confidence_accuracy": float(high_conf["correct"].mean()),
    }


def _print_confidence_summary(
    summary: dict, df: pd.DataFrame, y_true: np.ndarray, intensity_groups: bool
):
    """Print confidence analysis to console."""
    print("Confience Analysis:")
    print(f"    Correct:   mean conf = {summary['mean_confidence_correct']:.3f}")
    print(f"    Incorrect: mean conf = {summary['mean_confidence_incorrect']:.3f}")
    print(
        f"\nLow confidence (<{summary['threshold']}): {summary['low_confidence_count']} samples, acc = {summary['low_confidence_accuracy']:.3f}"
    )
    print(
        f"High confidence (>={summary['threshold']}): {summary['high_confidence_count']} samples, acc = {summary['high_confidence_accuracy']:.3f}"
    )

    print("\nBy Class:")
    for label in np.unique(y_true):
        class_df = df[df["true_label"] == label]
        label_name = get_class_labels(
            np.array([label]), intensity_groups=intensity_groups
        )[0]
        print(
            f"    {label_name:20s}: conf = {class_df['confidence'].mean():.3f}, acc = {class_df['correct'].mean():.3f}"
        )
    print("")


def _plot_confidence_analysis(df: pd.DataFrame, threshold: float, output_dir: Path):
    """Create confidence analysis plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].hist(
        df[df["correct"]]["confidence"],
        bins=20,
        alpha=0.7,
        label="Correct",
        color="green",
    )
    axes[0].hist(
        df[~df["correct"]]["confidence"],
        bins=20,
        alpha=0.7,
        label="Incorrect",
        color="red",
    )
    axes[0].axvline(
        threshold, color="black", linestyle="--", label=f"Threshold ({threshold})"
    )
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    for i in range(len(bins) - 1):
        bin_mask = (df["confidence"] >= bins[i]) & (df["confidence"] < bins[i + 1])
        bin_accuracies.append(
            df[bin_mask]["correct"].mean() if bin_mask.sum() > 0 else 0
        )

    axes[1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7)
    axes[1].axhline(
        df["correct"].mean(), color="green", linestyle="--", label="Overall accuracy"
    )
    axes[1].set_xlabel("Confidence Bin")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Confidence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
