import numpy as np


def extract_features_from_windows(windows: np.ndarray) -> np.ndarray:
    """Extract 1D features from window samples.
    
    Args:
        windows: Window samples to extract features from
        
    Returns:
        Extracted features corresponding to each window
    """
    n_windows, _, n_sensors = windows.shape

    features = np.zeros((n_windows, n_sensors * 4))
    for i in range(n_windows):
        window = windows[i]

        features[i] = np.concatenate([
            np.nanmean(window, axis=0),   # Ignore NaN
            np.nanstd(window, axis=0),    # Ignore NaN
            np.nanmin(window, axis=0),    # Ignore NaN
            np.nanmax(window, axis=0),    # Ignore NaN
        ])

    return features

