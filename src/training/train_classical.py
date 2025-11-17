from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import Config
from src.data import extract_features_from_windows, get_class_labels
from src.evaluation import evaluate_classical_model, save_evaluation_metrics
from src.models.classifiers import initialize_random_forest, initialize_xgboost


def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    sensor_cols: list = None,
    output_dir: Path = None,
) -> tuple:
    """Train and evaluate classifical ML model (Random Forest or XGBoost).

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Testing data
        cfg: Config object with experimental parameters
        output_dir: Directory to save results

    Returns:
        tuple: (model, label_encoder, results_dict)

    """
    X_train_feat = extract_features_from_windows(X_train)
    X_val_feat = extract_features_from_windows(X_val)
    X_test_feat = extract_features_from_windows(X_test)

    label_encoder = None
    if cfg.model_name == "xgboost":
        label_encoder = LabelEncoder()
        all_labels = np.concatenate([y_train, y_val, y_test])
        label_encoder.fit(all_labels)

        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_train_encoded = y_train
        y_val_encoded = y_val
        y_test_encoded = y_test

    if cfg.model_name == "random_forest":
        model = initialize_random_forest(cfg.random_forest_params, cfg.seed)
    elif cfg.model_name == "xgboost":
        model = initialize_xgboost(cfg.xgboost_params, cfg.seed)
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}")

    print(f"Training {cfg.model_name}")
    model.fit(X_train_feat, y_train_encoded)

    print("\nEvaluating model...")
    train_acc = model.score(X_train_feat, y_train_encoded)
    val_results = evaluate_classical_model(
        model, X_val_feat, y_val_encoded, label_encoder=label_encoder
    )
    test_results = evaluate_classical_model(
        model, X_test_feat, y_test_encoded, label_encoder=label_encoder
    )

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_results['accuracy']:.4f}")
    print(f"Test Accuracy:  {test_results['accuracy']:.4f}")

    if output_dir:
        if cfg.model_name == "random_forest" and sensor_cols:
            feature_names = []
            for sensor in sensor_cols:
                feature_names.extend(
                    [
                        f"{sensor}_mean",
                        f"{sensor}_std",
                        f"{sensor}_min",
                        f"{sensor}_max",
                    ]
                )

            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 8))
            top_20 = importance_df.head(20)
            ax.barh(range(20), top_20["importance"])
            ax.set_yticks(range(20))
            ax.set_yticklabels(top_20["feature"], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
            ax.set_title(
                "Top 20 Most Important Features", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(
                output_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        save_evaluation_metrics(
            val_results, output_dir, "val", cfg.model_name, cfg.group_activities
        )
        save_evaluation_metrics(
            test_results, output_dir, "test", cfg.model_name, cfg.group_activities
        )

    return (
        model,
        label_encoder,
        {
            "train_acc": train_acc,
            "val_acc": val_results["accuracy"],
            "test_acc": test_results["accuracy"],
            "test_classification_report": test_results["classification_report"],
        },
    )
