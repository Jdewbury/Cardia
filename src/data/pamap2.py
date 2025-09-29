import pandas as pd
from pathlib import Path

def get_column_names() -> list:
    """Initialize PAMAP2 column labels
    
    Returns:
        List of PAMAP2 data column names in appropriate order
    """
    imu_cols = [
        "temperature",
        "acc_x", "acc_y", "acc_z",
        "acc_x_2", "acc_y_2", "acc_z_2",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z",
        "orient_1", "orient_2", "orient_3", "orient_4",
    ]

    col_names = ["timestamp", "activity_id", "heart_rate"]

    for location in ["hand", "chest", "ankle"]:
        for c in imu_cols:
            col_names.append(f"{location}_{c}")
            
    return col_names

def _filter_to_chest(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only contain chest IMU labels

    Args:
        df: dataframe object to filter for chest labels
    
    Returns:
        Filtered dataframe with chest labels
    """
    base_cols = ["timestamp", "activity_id", "heart_rate", "subject_id"]
    chest_cols = [c for c in df.columns if c.startswith("chest_")]
    cols_to_keep = base_cols + chest_cols
    return df[cols_to_keep]

def load_pamap2(data_dir: Path, filter_chest: bool = True) -> pd.DataFrame:
    """Load PAMAP2 patient dataset

    Args:
        data_dir: directory containing patient data
        filter_chest: whether to filter for only chest IMU labels
    
    Returns:
        Dataframe object of PAMAP2 patients
    """
    col_names = get_column_names()
    
    all_dfs = []
    for file_path in sorted(data_dir.glob("*.dat")):
        df = pd.read_csv(file_path, delimiter=" ", header=None, names=col_names)
        df["subject_id"] = file_path.stem.replace("subject", "")
        all_dfs.append(df)
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    if filter_chest:
        combined_df = _filter_to_chest(combined_df)
    
    return combined_df