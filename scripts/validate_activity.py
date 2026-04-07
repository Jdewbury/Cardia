import warnings
from pathlib import Path

import joblib
import numpy as np

from src.config import Config
from src.data import load_recording
from src.data.constants import PAMAP2_ACTIVITY_MAPPING
from src.evaluation import predict_interval

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    cfg = Config()
    cfg.update_from_args()

    model = joblib.load(cfg.model_path)

    df_converted = load_recording(
        cfg.recording_path,
        use_hr=cfg.use_heart_rate,
        sensor_orientation=cfg.sensor_orientation,
        ecg_channel=cfg.ecg_channel,
    )

    print(f"Recording: {Path(cfg.recording_path).parent.name}")
    print(f"Duration: {df_converted['timestamp'].max():.1f}s")
    print(f"Smoothing: {'Enabled' if cfg.smoothing else 'Disabled'}\n")

    interval_list = [
        tuple(interval.split("-")) for interval in cfg.intervals.split(",")
    ]

    for start, end in interval_list:
        predictions, confidences = predict_interval(
            df_converted,
            start,
            end,
            model,
            use_hr=cfg.use_heart_rate,
            use_smoothing=cfg.smoothing,
        )

        if predictions is None:
            print(f"[{start}-{end}]: No data\n")
            continue

        print(f"[{start}-{end}] ({len(predictions)} windows):")
        unique, counts = np.unique(predictions, return_counts=True)
        for activity_id, count in zip(unique, counts):
            activity_label = PAMAP2_ACTIVITY_MAPPING.get(
                activity_id, f"Unknown_{activity_id}"
            )
            pct = count / len(predictions) * 100
            avg_conf = confidences[predictions == activity_id].mean()
            print(f"  {activity_label:20s}: {pct:5.1f}% (conf: {avg_conf:.2f})")
        print()


if __name__ == "__main__":
    main()
