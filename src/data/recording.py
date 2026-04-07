import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


def extract_hr_from_ecg(ecg_signal: np.ndarray, timestamps: np.ndarray, f_s: int = 125) -> pd.DataFrame:
    """Extract heart rate from ECG signal using Pan-Tompkins algorithm.
    
    Args:
        ecg_signal: Raw ECG signal
        timestamps: Timestamp array
        f_s: Sampling frequency in Hz
        
    Returns:
        DataFrame with timestamps and sparse heart rate values
    """
    b, a = butter(2, [5, 20], btype="band", fs=f_s)
    filtered = filtfilt(b, a, ecg_signal)
    
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    
    N = int(0.15 * f_s)
    integrated = np.convolve(squared, np.ones(N) / N, mode='same')
    
    r_peaks, _ = find_peaks(
        integrated,
        distance=int(0.3 * f_s),
        prominence=np.percentile(integrated, 50)
    )
    
    if len(r_peaks) < 2:
        return pd.DataFrame({'timestamp': timestamps, 'heart_rate': np.nan})
    
    rr_intervals = np.diff(timestamps[r_peaks])
    hr_values = 60.0 / rr_intervals
    hr_timestamps = timestamps[r_peaks[1:]]
    
    valid = (hr_values >= 40) & (hr_values <= 200)
    hr_values = hr_values[valid]
    hr_timestamps = hr_timestamps[valid]
    
    hr_df = pd.DataFrame({'timestamp': hr_timestamps, 'heart_rate': hr_values})
    full_df = pd.DataFrame({'timestamp': timestamps})
    full_df = full_df.merge(hr_df, on='timestamp', how='left')
    
    return full_df


def convert_to_pamap2_format(df_raw: pd.DataFrame, use_hr: bool = False, 
                             sensor_orientation: str = "new", 
                             ecg_channel: str = "ecg_leadIII") -> pd.DataFrame:
    """Convert raw recording to PAMAP2 format.
    
    Args:
        df_raw: Raw dataframe with IMU and ECG data
        use_hr: Whether to extract heart rate from ECG
        sensor_orientation: "old" or "new" sensor mounting configuration
        ecg_channel: ECG channel name to use
        
    Returns:
        DataFrame in PAMAP2 format with chest IMU and optional HR
    """
    df = pd.DataFrame()
    accel_scale = (8 * 9.8) / 32768
    gyro_scale = (2000 * np.pi / 180) / 32768
    
    chest_ax = pd.to_numeric(df_raw["chest_ax"], errors='coerce')
    chest_ay = pd.to_numeric(df_raw["chest_ay"], errors='coerce')
    chest_az = pd.to_numeric(df_raw["chest_az"], errors='coerce')
    chest_gx = pd.to_numeric(df_raw["chest_gx"], errors='coerce')
    chest_gy = pd.to_numeric(df_raw["chest_gy"], errors='coerce')
    chest_gz = pd.to_numeric(df_raw["chest_gz"], errors='coerce')
    
    chest_ax = chest_ax.where(chest_ax.abs() <= 8000, np.nan)
    chest_ay = chest_ay.where(chest_ay.abs() <= 8000, np.nan)
    chest_az = chest_az.where(chest_az.abs() <= 8000, np.nan)
    
    if sensor_orientation == "old":
        df["chest_acc_x"] = chest_ay * accel_scale
        df["chest_acc_y"] = chest_ax * accel_scale
        df["chest_acc_z"] = chest_az * accel_scale
        df["chest_gyro_x"] = chest_gy * gyro_scale
        df["chest_gyro_y"] = chest_gx * gyro_scale
        df["chest_gyro_z"] = chest_gz * gyro_scale
    else:
        df["chest_acc_x"] = chest_ax * accel_scale
        df["chest_acc_y"] = -chest_ay * accel_scale
        df["chest_acc_z"] = -chest_az * accel_scale
        df["chest_gyro_x"] = chest_gx * gyro_scale
        df["chest_gyro_y"] = -chest_gy * gyro_scale
        df["chest_gyro_z"] = chest_gz * gyro_scale
    
    timestamps_dt = pd.to_datetime(df_raw["timestamp"], format='mixed')
    df["timestamp_dt"] = timestamps_dt
    df["timestamp"] = (timestamps_dt - timestamps_dt.iloc[0]).dt.total_seconds()
    
    df = df.dropna(subset=["chest_acc_x", "chest_acc_y", "chest_acc_z"])
    
    if use_hr:
        ecg_signal = pd.to_numeric(df_raw.loc[df.index, ecg_channel], errors='coerce').values
        hr_df = extract_hr_from_ecg(ecg_signal, df["timestamp"].values)
        df["heart_rate"] = hr_df["heart_rate"].values
    
    return df


def load_recording(recording_path: str, use_hr: bool = False,
                  sensor_orientation: str = "new", 
                  ecg_channel: str = "ecg_leadIII") -> pd.DataFrame:
    """Load and convert a raw recording file.
    
    Args:
        recording_path: Path to CSV recording file
        use_hr: Whether to extract heart rate
        sensor_orientation: Sensor mounting configuration
        ecg_channel: ECG channel to use
        
    Returns:
        Converted dataframe in PAMAP2 format
    """
    columns = ['timestamp', ecg_channel, 'chest_ax', 'chest_ay', 'chest_az', 
               'chest_gx', 'chest_gy', 'chest_gz']
    
    df_raw = pd.read_csv(recording_path, low_memory=False, usecols=columns)
    df_converted = convert_to_pamap2_format(
        df_raw, 
        use_hr=use_hr, 
        sensor_orientation=sensor_orientation,
        ecg_channel=ecg_channel
    )
    
    return df_converted