from pathlib import Path

from src.config import Config
from src.data import filter_common_activities, load_pamap2, split_subjects
from src.data.constants import map_to_intensity_groups
from src.training import sweep_window_length
from src.utils import make_dir, set_all_seeds


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
            df_clean["activity_id"].values
        )
        print(f"Converted to {len(df_clean['activity_id'].unique())} intensity groups")

    train_df, val_df, test_df = split_subjects(df_clean)
    train_df, val_df, test_df = filter_common_activities(train_df, val_df, test_df)

    sensor_cols = [col for col in train_df.columns if col.startswith("chest_")]

    if cfg.use_heart_rate and "heart_rate" in train_df.columns:
        sensor_cols.append("heart_rate")

    exp_dir = make_dir(cfg.experiment_dir)
    cfg.save(exp_dir / "config.json")
    print(f"Config saved to: {exp_dir / 'config.json'}")

    sweep_window_length(
        train_df,
        val_df,
        sensor_cols,
        cfg,
        output_dir=exp_dir,
    )


if __name__ == "__main__":
    main()
