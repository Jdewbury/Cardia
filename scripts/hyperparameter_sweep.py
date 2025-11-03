from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import Config
from src.data import (
    extract_features_from_windows,
    filter_common_activities,
    get_windows,
    load_pamap2,
    split_subjects,
)
from src.data.constants import map_to_intensity_groups
from src.training import sweep_random_forest_hyperparams, sweep_xgboost_hyperparams
from src.utils import make_dir, set_all_seeds


def main():
    cfg = Config()
    cfg.update_from_args()
    cfg.window_size_sec = 10.0  # set constant for sweep

    set_all_seeds(cfg.seed)

    df = load_pamap2(Path(cfg.data_dir), filter_chest=cfg.filter_chest)
    df_clean = df[df["activity_id"] != 0].copy()

    if cfg.group_activities:
        df_clean["activity_id"] = map_to_intensity_groups(
            df_clean["activity_id"].values
        )
        print(f"Converted to {len(df_clean['activity_id'].unique())} intensity groups")

    train_df, val_df, test_df = split_subjects(df_clean)
    train_df, val_df, test_df = filter_common_activities(train_df, val_df, test_df)

    sensor_cols = [col for col in train_df.columns if col.startswith("chest_")]

    X_train, y_train = get_windows(
        train_df, sensor_cols, cfg.window_size_samples, cfg.stride
    )
    X_val, y_val = get_windows(val_df, sensor_cols, cfg.window_size_samples, cfg.stride)
    print(f"Train:  {X_train.shape[0]} windows")
    print(f"Val:    {X_val.shape[0]} windows")

    X_train_feat = extract_features_from_windows(X_train)
    X_val_feat = extract_features_from_windows(X_val)

    exp_dir = make_dir(cfg.experiment_dir)
    cfg.save(exp_dir / "config.json")
    print(f"Config saved to: {exp_dir / 'config.json'}")

    if cfg.model_name == "random_forest":
        results, params, model = sweep_random_forest_hyperparams(
            X_train_feat, y_train, X_val_feat, y_val, output_dir=exp_dir
        )
    elif cfg.model_name == "xgboost":
        label_encoder = LabelEncoder()
        all_labels = np.concatenate([y_train, y_val])
        label_encoder.fit(all_labels)

        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        results, params, model = sweep_xgboost_hyperparams(
            X_train_feat, y_train_encoded, X_val_feat, y_val_encoded, output_dir=exp_dir
        )
    else:
        raise ValueError(f"Invalid model name: {cfg.model_name}")


if __name__ == "__main__":
    main()
