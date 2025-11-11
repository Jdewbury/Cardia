from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

from src.config import Config
from src.data import extract_features_from_windows, get_windows
from src.utils import make_dir, save_file


def sweep_random_forest_hyperparams(
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


def sweep_xgboost_hyperparams(
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
                "train_accuracy": train_acc,
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

    print("\nBest XGBoost")
    print(f"    Val Accuracy: {best_row.val_accuracy:.4f}")
    print(f"    Params: {best_params}")

    return results_df, best_params, best_model


def sweep_window_length(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sensor_cols: list,
    cfg: Config,
    output_dir: str | Path = None,
    maximum_length: int = 30,
) -> tuple:
    """Sweep model over different window lengths (seconds).

    Args:
        train_df, val_df: Dataframe containing patient samples
        sensor_cols: Name of sensor columns
        cfg: Initialized Config object containing experimental parameters
        output_dir: Desired directory to save hyperparameter results (None = do not save)

    Returns:
        tuple: (results_df, best_window_size_sec)
    """
    window_configs = np.arange(1, maximum_length + 1, dtype=float)

    if cfg.model_name == "xgboost":
        label_encoder = LabelEncoder()
        all_labels = np.concatenate(
            [train_df.activity_id.unique(), val_df.activity_id.unique()]
        )
        label_encoder.fit(all_labels)

    results = []
    for window_size_sec in tqdm(window_configs, desc="Running window size sweep"):
        cfg.window_size_sec = window_size_sec

        X_train, y_train = get_windows(
            train_df, sensor_cols, cfg.window_size_samples, cfg.stride
        )
        X_val, y_val = get_windows(
            val_df, sensor_cols, cfg.window_size_samples, cfg.stride
        )

        X_train_feat = extract_features_from_windows(X_train)
        X_val_feat = extract_features_from_windows(X_val)

        if cfg.model_name == "xgboost":
            y_train_encoded = label_encoder.transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val

        if cfg.model_name == "random_forest":
            model_params = cfg.random_forest_params
            model = RandomForestClassifier(
                **model_params, random_state=cfg.seed, n_jobs=-1
            )
        elif cfg.model_name == "xgboost":
            model_params = cfg.xgboost_params
            model = XGBClassifier(
                **model_params, random_state=cfg.seed, n_jobs=-1, eval_metric="mlogloss"
            )
        else:
            raise ValueError(f"Invalid model name: {cfg.model_name}")

        model.fit(X_train_feat, y_train_encoded)
        train_acc = model.score(X_train_feat, y_train_encoded)
        val_acc = model.score(X_val_feat, y_val_encoded)

        results.append(
            {
                "window_size_sec": window_size_sec,
                "window_size_samples": cfg.window_size_samples,
                "n_train_windows": len(X_train),
                "n_val_windows": len(X_val),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

    results_df = pd.DataFrame(results)

    best_idx = results_df.val_accuracy.idxmax()
    best_row = results_df.iloc[best_idx]

    if output_dir:
        output_dir = make_dir(output_dir)

        results_df.to_csv(
            output_dir / f"{cfg.model_name}_window_sweep_results.csv", index=False
        )

        best_config = {
            "model": f"{type(model).__name__}",
            "model_params": model_params,
            "best_window_size_sec": float(best_row.window_size_sec),
            "best_window_size_samples": int(best_row.window_size_samples),
            "performance": {
                "train_accuracy": float(best_row.train_accuracy),
                "val_accuracy": float(best_row.val_accuracy),
            },
            "window_configs_tested": window_configs.tolist(),
        }
        save_file(output_dir / f"{cfg.model_name}_window_best_params.json", best_config)

        print(f"Results saved to: {output_dir}")

    print(f"\nBest {cfg.model_name}")
    print(f"    Val Accuracy: {best_row.val_accuracy:.4f}")
    print(f"    Params: {best_row.window_size_sec}")

    return results_df, float(best_row.window_size_sec)


def sweep_sampling_rate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sensor_cols: list,
    cfg: Config,
    output_dir: str | Path = None,
) -> tuple:
    """Sweep model over different sampling rates (Hz).

    Args:
        train_df, val_df: Dataframe containing patient samples
        sensor_cols: Name of sensor columns
        cfg: Initialized Config object containing experimental parameters
        output_dir: Desired directory to save hyperparameter results (None = do not save)

    Returns:
        tuple: (results_df, best_sampling_rate)
    """
    sampling_rates = [100, 50, 25, 20, 10, 5, 2, 1]
    original_sampling_rate = cfg.sampling_rate

    if cfg.model_name == "xgboost":
        label_encoder = LabelEncoder()
        all_labels = np.concatenate(
            [train_df.activity_id.unique(), val_df.activity_id.unique()]
        )
        label_encoder.fit(all_labels)

    results = []
    for sampling_rate in tqdm(sampling_rates, desc="Running window size sweep"):
        cfg.sampling_rate = sampling_rate
        downsample_factor = int(original_sampling_rate / sampling_rate)

        train_df_downsampled = train_df.iloc[::downsample_factor].reset_index(drop=True)
        val_df_downsampled = val_df.iloc[::downsample_factor].reset_index(drop=True)

        X_train, y_train = get_windows(
            train_df_downsampled, sensor_cols, cfg.window_size_samples, cfg.stride
        )
        X_val, y_val = get_windows(
            val_df_downsampled, sensor_cols, cfg.window_size_samples, cfg.stride
        )

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Warning: No windows generated for {sampling_rate} Hz, skipping...")
            continue

        X_train_feat = extract_features_from_windows(X_train)
        X_val_feat = extract_features_from_windows(X_val)

        if cfg.model_name == "xgboost":
            y_train_encoded = label_encoder.transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val

        if cfg.model_name == "random_forest":
            model_params = cfg.random_forest_params
            model = RandomForestClassifier(
                **model_params, random_state=cfg.seed, n_jobs=-1
            )
        elif cfg.model_name == "xgboost":
            model_params = cfg.xgboost_params
            model = XGBClassifier(
                **model_params, random_state=cfg.seed, n_jobs=-1, eval_metric="mlogloss"
            )
        else:
            raise ValueError(f"Invalid model name: {cfg.model_name}")

        model.fit(X_train_feat, y_train_encoded)
        train_acc = model.score(X_train_feat, y_train_encoded)
        val_acc = model.score(X_val_feat, y_val_encoded)

        results.append(
            {
                "sampling_rate": sampling_rate,
                "downsample_factor": downsample_factor,
                "n_train_windows": len(X_train),
                "n_val_windows": len(X_val),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

    results_df = pd.DataFrame(results)

    best_idx = results_df.val_accuracy.idxmax()
    best_row = results_df.iloc[best_idx]

    if output_dir:
        output_dir = make_dir(output_dir)

        results_df.to_csv(
            output_dir / f"{cfg.model_name}_sampling_rate_results.csv", index=False
        )

        best_config = {
            "model": f"{type(model).__name__}",
            "model_params": model_params,
            "best_sampling_rate": int(best_row.sampling_rate),
            "best_downsample_factor": int(best_row.downsample_factor),
            "performance": {
                "train_accuracy": float(best_row.train_accuracy),
                "val_accuracy": float(best_row.val_accuracy),
            },
            "sampling_rates_tested": sampling_rates,
        }
        save_file(
            output_dir / f"{cfg.model_name}_sampling_rate_best_params.json", best_config
        )

        print(f"Results saved to: {output_dir}")

    print(f"\nBest {cfg.model_name}")
    print(f"    Val Accuracy: {best_row.val_accuracy:.4f}")
    print(f"    Sampling Rate: {best_row.sampling_rate} Hz")

    return results_df, int(best_row.sampling_rate)
