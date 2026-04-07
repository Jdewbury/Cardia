import numpy as np
import pandas as pd

from src.data import extract_features_from_windows


def apply_minimum_duration_smoothing(
    predictions: np.ndarray, min_windows: int = 3
) -> np.ndarray:
    """Apply minimum duration post-processing to remove short activity segments.

    Args:
        predictions: Array of activity predictions
        min_windows: Minimum number of windows for valid segment

    Returns:
        Smoothed predictions array
    """
    predictions = predictions.astype(np.int32)
    smoothed = predictions.copy()

    changes = np.where(np.diff(predictions) != 0)[0] + 1
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [len(predictions)]])

    for start, end in zip(segment_starts, segment_ends):
        if end - start < min_windows:
            before = predictions[max(0, start - 3) : start]
            after = predictions[end : min(len(predictions), end + 3)]
            surrounding = np.concatenate([before, after])

            if len(surrounding) > 0:
                replacement = np.bincount(surrounding.astype(np.int32)).argmax()
                smoothed[start:end] = replacement

    return smoothed


def predict_interval(
    df_converted: pd.DataFrame,
    start_time: str,
    end_time: str,
    model,
    use_hr: bool = False,
    use_smoothing: bool = True,
) -> tuple:
    """Predict activity for a time interval.

    Args:
        df_converted: Converted dataframe in PAMAP2 format
        start_time: Start time string
        end_time: End time string
        model: Trained classifier model
        use_hr: Whether to use heart rate features
        use_smoothing: Whether to apply smoothing post-processing

    Returns:
        Tuple of (predictions, probabilities) or (None, None) if no data
    """
    sensor_cols = [
        "chest_acc_x",
        "chest_acc_y",
        "chest_acc_z",
        "chest_gyro_x",
        "chest_gyro_y",
        "chest_gyro_z",
    ]
    if use_hr:
        sensor_cols.append("heart_rate")

    start_dt = pd.to_datetime(start_time, format="mixed")
    end_dt = pd.to_datetime(end_time, format="mixed")
    mask = (df_converted["timestamp_dt"] >= start_dt) & (
        df_converted["timestamp_dt"] <= end_dt
    )
    activity_df = df_converted[mask].reset_index(drop=True)

    windows = []
    start_t = activity_df["timestamp"].min()
    end_t = activity_df["timestamp"].max()
    current_time = start_t

    while current_time + 10.0 <= end_t:
        mask_w = (activity_df["timestamp"] >= current_time) & (
            activity_df["timestamp"] < current_time + 10.0
        )
        window_data = activity_df.loc[mask_w, sensor_cols].values
        if len(window_data) > 0:
            windows.append(window_data)
        current_time += 5.0

    if len(windows) == 0:
        return None, None

    max_len = max(len(w) for w in windows)
    X_windows = np.full((len(windows), max_len, len(sensor_cols)), np.nan)
    for i, window in enumerate(windows):
        X_windows[i, : len(window), :] = window

    X_features = extract_features_from_windows(X_windows)
    predictions_raw = model.predict(X_features)
    probabilities = model.predict_proba(X_features)
    confidences_raw = probabilities.max(axis=1)

    if use_smoothing:
        predictions = apply_minimum_duration_smoothing(predictions_raw, min_windows=3)
    else:
        predictions = predictions_raw

    segment_confidences = np.zeros_like(predictions, dtype=float)

    for activity_id in np.unique(predictions):
        mask = predictions == activity_id
        segment_confidences[mask] = confidences_raw[mask].mean()

    return predictions, segment_confidences
