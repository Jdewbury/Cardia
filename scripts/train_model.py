from pathlib import Path

import joblib
import pandas as pd

from src.config import Config
from src.data import filter_common_activities, get_windows, load_pamap2, split_subjects
from src.data.constants import map_to_intensity_groups
from src.training.train_classical import train_classical_model
from src.utils import make_dir, save_file, set_all_seeds


def main():
    cfg = Config()
    cfg.update_from_args()

    set_all_seeds(cfg.seed)

    df = load_pamap2(
        Path(cfg.data_dir),
        filter_chest=cfg.filter_chest,
        exclude_sensors=cfg.exclude_sensors,
        combine_similar=cfg.combine_similar,
    )
    df_clean = df[df["activity_id"] != 0]

    if cfg.group_activities:
        df_clean["activity_id"] = map_to_intensity_groups(
            df_clean["activity_id"].astype(int).values
        )
        print(f"Converted to {len(df_clean['activity_id'].unique())} intensity groups")

    train_df, val_df, test_df = split_subjects(df_clean)
    train_df, val_df, test_df = filter_common_activities(train_df, val_df, test_df)

    if cfg.use_all_data:
        print("Combining all data for training")
        train_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        val_df = None
        test_df = None

    sensor_cols = [col for col in train_df.columns if col.startswith("chest_")]

    if cfg.use_heart_rate and "heart_rate" in train_df.columns:
        sensor_cols.append("heart_rate")

    X_train, y_train = get_windows(
        train_df, sensor_cols, cfg.window_size_samples, cfg.stride
    )

    if val_df is not None:
        X_val, y_val = get_windows(
            val_df, sensor_cols, cfg.window_size_samples, cfg.stride
        )
    else:
        X_val, y_val = None, None

    if test_df is not None:
        X_test, y_test = get_windows(
            test_df, sensor_cols, cfg.window_size_samples, cfg.stride
        )
    else:
        X_test, y_test = None, None

    exp_dir = make_dir(cfg.experiment_dir)
    if cfg.model_name in ["random_forest", "xgboost"]:
        model, label_encoder, metrics = train_classical_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
            sensor_cols,
            output_dir=exp_dir,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_name}")

    exp_dir = make_dir(cfg.experiment_dir)
    cfg.save(exp_dir / "config.json")

    if cfg.model_name in ["random_forest", "xgboost"]:
        model_path = exp_dir / f"{cfg.model_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

    if label_encoder is not None:
        encoder_path = exp_dir / "label_encoder.pkl"
        joblib.dump(label_encoder, encoder_path)

    results = {
        "model": cfg.model_name,
        "window_size_sec": cfg.window_size_sec,
        "performance": metrics,
    }
    save_file(exp_dir / "results.json", results)
    print(f"Results saved to: {exp_dir / 'results.json'}")


if __name__ == "__main__":
    main()
