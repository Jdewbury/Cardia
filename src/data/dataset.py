import numpy as np
import pandas as pd


def get_windows(
    df: pd.DataFrame, sensor_cols: list, window_size: int, stride: int
) -> tuple:
    """Extract windows from time-series data.
    
    Args:
        df: Dataframe containing time-series samples
        sensor_cols: Name of sensor columns
        window_size: Desired window size (in indices)
        stride: Desired stride (in indices)
        
    Returns:
        tuple: (windows, labels)
        
    """
    windows = []
    labels = []

    for (subject, activity), group in df.groupby(["subject_id", "activity_id"]):
        group = group.sort_values("timestamp").reset_index(drop=True)

        sensor_data = group[sensor_cols].values

        for start_idx in range(0, len(sensor_data) - window_size + 1, stride):
            window = sensor_data[start_idx : start_idx + window_size]

            if not np.isnan(window).any():
                windows.append(window)
                labels.append(activity)

    return np.array(windows), np.array(labels)


def split_subjects(df: pd.DataFrame) -> tuple:
    """Split subjects into sets.
    
    Args: 
        df: Dataframe containing time-series samples
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    subject_ids = sorted(df.subject_id.unique())
    train_val_subjects, test_subjects = np.split(subject_ids, [-2])

    np.random.shuffle(train_val_subjects)
    train_subjects, val_subjects = np.split(train_val_subjects, [-2])

    train_df = df[df.subject_id.isin(train_subjects)].copy()
    val_df = df[df.subject_id.isin(val_subjects)].copy()
    test_df = df[df.subject_id.isin(test_subjects)].copy()

    return train_df, val_df, test_df


def filter_common_activities(
    *dfs: pd.DataFrame, return_activities: bool = False
) -> tuple | tuple[tuple, list]:
    """Filter data sets by common activities.
    
    Args:
        *dfs: Unique dataframes to filter for common activities
        return_activities: Flag to return the common activities
    
    Returns:
        tuple: (filtered_dfs), common_activities (optional)
    """
    common_activities = set(dfs[0].activity_id.unique())
    for df in dfs[1:]:
        common_activities &= set(df.activity_id.unique())

    filtered_dfs = []
    for df in dfs:
        filtered_df = df[df.activity_id.isin(common_activities)].copy()
        filtered_dfs.append(filtered_df)

    if return_activities:
        return tuple(filtered_dfs), common_activities
    
    return tuple(filtered_dfs)
