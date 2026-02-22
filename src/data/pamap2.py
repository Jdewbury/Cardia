from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import remap_similar_activities


def get_column_names() -> list:
    """Initialize PAMAP2 column labels.
    
    Returns:
        List of PAMAP2 data column names in appropriate order
    """
    imu_cols = [
        "temperature",
        "acc_x",
        "acc_y",
        "acc_z",
        "acc_x_2",
        "acc_y_2",
        "acc_z_2",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
        "orient_1",
        "orient_2",
        "orient_3",
        "orient_4",
    ]

    col_names = ["timestamp", "activity_id", "heart_rate"]

    for location in ["hand", "chest", "ankle"]:
        for c in imu_cols:
            col_names.append(f"{location}_{c}")

    return col_names


def _filter_to_chest(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only contain chest IMU labels.

    Args:
        df: Dataframe object to filter for chest labels
    
    Returns:
        Filtered dataframe with chest labels
    """
    base_cols = ["timestamp", "activity_id", "heart_rate", "subject_id"]
    chest_cols = [c for c in df.columns if c.startswith("chest_")]
    cols_to_keep = base_cols + chest_cols
    return df[cols_to_keep]


def load_pamap2(
    data_dir: Path, 
    filter_chest: bool = True, 
    exclude_sensors: list = None,
    combine_similar: bool = False
) -> pd.DataFrame:
    col_names = get_column_names()
    
    if filter_chest:
        base_cols = ["timestamp", "activity_id", "heart_rate"]
        chest_cols = [c for c in col_names if c.startswith("chest_")]
        if exclude_sensors:
            chest_cols = [c for c in chest_cols if c.replace("chest_", "") not in exclude_sensors]
        cols_to_use = base_cols + chest_cols
        usecols = [col_names.index(c) for c in cols_to_use]
    else:
        cols_to_use = col_names
        usecols = list(range(len(col_names)))

    all_arrays = []
    subject_ids = []
    
    for file_path in tqdm(sorted(data_dir.glob("*.dat")), desc="Loading PAMAP2 patient data"):
        data = np.loadtxt(file_path, usecols=usecols, dtype=np.float32)
        all_arrays.append(data)
        
        s_id = file_path.stem.replace("subject", "")
        subject_ids.extend([s_id] * len(data))
        del data

    combined = np.vstack(all_arrays)
    del all_arrays
    
    combined_df = pd.DataFrame(combined, columns=cols_to_use)
    combined_df["subject_id"] = subject_ids
    del combined
    del subject_ids
    
    if combine_similar:
        combined_df = remap_similar_activities(combined_df)
        print("Combined similar activities")
    if exclude_sensors:
        print(f"Excluded sensors: {exclude_sensors}")
    
    return combined_df
