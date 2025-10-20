from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from src.utils import make_dir, save_file


def sweep_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str | Path = None,
) -> tuple:
    """Performs hyperparameter sweep on Random Forest model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        output_dir: Desired directory to save hyperparameter results (None = do not save)

    Returns:
        tuple: (results, best_params, best_model)

    """
    param_grid = {
        "n_estimators": [25, 50, 75, 100, 150, 200],
        "max_depth": [15, 20, 25, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    total_combos = np.prod([len(v) for v in param_grid.values()])

    results = []
    for params in tqdm(
        product(
            param_grid["n_estimators"],
            param_grid["max_depth"],
            param_grid["min_samples_split"],
            param_grid["min_samples_leaf"],
        ),
        total=total_combos,
        desc="Running RF hyperparameter sweep",
    ):
        n_est, depth, split, leaf = params

        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf,
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)
        train_acc = rf.score(X_train, y_train)
        val_acc = rf.score(X_val, y_val)

        results.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "min_samples_split": split,
                "min_samples_leaf": leaf,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

    results_df = pd.DataFrame(results)

    best_idx = results_df.val_accuracy.idxmax()
    best_row = results_df.iloc[best_idx]

    best_params = {
        "n_estimators": int(best_row.n_estimators),
        "max_depth": None if pd.isna(best_row.max_depth) else int(best_row.max_depth),
        "min_samples_split": int(best_row.min_samples_split),
        "min_samples_leaf": int(best_row.min_samples_leaf),
    }

    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    if output_dir:
        output_dir = make_dir(output_dir)

        results_df.to_csv(output_dir / "rf_sweep_results.csv", index=False)

        best_config = {
            "model": "RandomForestClassifier",
            "best_params": best_params,
            "performance": {
                "train_accuracy": float(best_row.train_accuracy),
                "val_accuracy": float(best_row.val_accuracy),
            },
            "param_grid": param_grid,
        }
        save_file(output_dir / "rf_best_params.json", best_config)

        print(f"Results saved to: {output_dir}")

    print("Best Random Forest")
    print(f"    Val Accuracy: {best_row.val_accuracy:.4f}")
    print(f"    Params: {best_params}")

    return results_df, best_params, best_model


def sweep_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: str | Path = None,
) -> tuple:
    """Performs hyperparameter sweep on XGBoost model.

    Args:
        X_train, y_train: Training data (labels must be encoded 0-N)
        X_val, y_val: Validation data (labels must be encoded 0-N)
        output_dir: Desired directory to save hyperparameter results (None = do not save)

    Returns:
        tuple: (results, best_params, best_model)

    """
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    total_combos = np.prod([len(v) for v in param_grid.values()])

    results = []
    for params in tqdm(
        product(
            param_grid["n_estimators"],
            param_grid["max_depth"],
            param_grid["learning_rate"],
            param_grid["subsample"],
            param_grid["colsample_bytree"],
        ),
        total=total_combos,
        desc="Running XGBoost hyperparameter sweep",
    ):
        n_est, depth, lr, subsample, colsample = params

        xgb = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

        xgb.fit(X_train, y_train)
        train_acc = xgb.score(X_train, y_train)
        val_acc = xgb.score(X_val, y_val)

        results.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": subsample,
                "colsample_bytree": colsample,
                "val_accuracy": val_acc,
            }
        )

    results_df = pd.DataFrame(results)

    best_idx = results_df.val_accuracy.idxmax()
    best_row = results_df.iloc[best_idx]

    best_params = {
        "n_estimators": int(best_row.n_estimators),
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row.learning_rate),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
    }

    best_model = XGBClassifier(
        **best_params, random_state=42, n_jobs=-1, eval_metric="mlogloss"
    )
    best_model.fit(X_train, y_train)

    if output_dir:
        output_dir = make_dir(output_dir)

        results_df.to_csv(output_dir / "xgb_sweep_results.csv", index=False)

        best_config = {
            "model": "XGBClassifier",
            "best_params": best_params,
            "performance": {
                "train_accuracy": float(best_row.train_accuracy),
                "val_accuracy": float(best_row.val_accuracy),
            },
            "param_grid": param_grid,
        }
        save_file(output_dir / "xgb_best_params.json", best_config)

        print(f"Results saved to: {output_dir}")

    print("Best XGBoost")
    print(f"    Val Accuracy: {best_row.val_accuracy:.4f}")
    print(f"    Params: {best_params}")

    return results_df, best_params, best_model
