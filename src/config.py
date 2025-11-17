import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.utils import load_file, save_file


@dataclass
class Config:
    # data
    data_dir: str = (
        r"data\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\Protocol"
    )
    dataset: str = "pamap2"
    filter_chest: bool = True
    exclude_sensors: list = None
    group_activities: bool = False
    data_sampling_rate: int = 100
    desired_sampling_rate: int = 100

    # preprocessing
    window_size_sec: float = 5.0
    overlap: float = 0.5
    sampling_rate: int = 100

    # model
    model_name: str = "random_forest"
    learning_rate: float = 0.1
    n_estimators: int = 50
    max_depth: int = 25

    # rf hyperparameters
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    # xgb hyperparameters
    subsample: float = 0.7
    colsample_bytree: float = 0.8

    # training
    save_model: bool = True
    save_predictions: bool = False
    verbose: bool = True

    # output
    output_dir: str = "experiments"
    experiment_name: str = None

    # other
    seed: int = 42

    def save(self, file_path: Path) -> None:
        save_file(file_path, asdict(self))

    @classmethod
    def load(cls, file_path: Path) -> "Config":
        data = load_file(file_path)
        return cls(**data)

    @property
    def experiment_dir(self) -> Path:
        if self.experiment_name:
            return Path(self.output_dir) / self.experiment_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.model_name}_w{self.window_size_sec}s_{timestamp}"

        return Path(self.output_dir) / name

    @property
    def window_size_samples(self) -> int:
        return int(self.window_size_sec * self.sampling_rate)

    @property
    def stride(self) -> int:
        return int(self.window_size_samples * (1 - self.overlap))

    @property
    def random_forest_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }

    @property
    def xgboost_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }

    def update_from_args(self) -> None:
        parser = argparse.ArgumentParser(description="Cardia Experimental Config")

        # data args
        parser.add_argument(
            "--data_dir",
            type=str,
            help=f"Path to data directory. Default: {self.data_dir}",
        )
        parser.add_argument(
            "--dataset", type=str, help=f"Dataset name. Default: {self.dataset}"
        )
        parser.add_argument(
            "--filter_chest",
            type=str,
            choices=["true", "false"],
            help=f"Filter chest sensors. Default: {self.filter_chest}",
        )
        parser.add_argument(
            "--exclude_sensors",
            nargs="+",
            default=None,
            help="Sensor types to exclude (e.g., mag_x acc_x_2 acc_y_2 acc_z_2)",
        )
        parser.add_argument(
            "--group_activities",
            type=str,
            choices=["true", "false"],
            help=f"Group activities by intensity. Default: {self.group_activities}",
        )

        # preprocessing args
        parser.add_argument(
            "--window_size_sec",
            type=float,
            help=f"Window size in seconds. Default: {self.window_size_sec}",
        )
        parser.add_argument(
            "--overlap",
            type=float,
            help=f"Window overlap fraction. Default: {self.overlap}",
        )
        parser.add_argument(
            "--sampling_rate",
            type=int,
            help=f"Sampling rate. Default: {self.sampling_rate}",
        )

        # model args
        parser.add_argument(
            "--model_name",
            type=str,
            choices=["random_forest", "xgboost"],
            help=f"Model name. Default: {self.model_name}",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            help=f"Learning rate. Default: {self.learning_rate}",
        )

        # rf hyperparams
        parser.add_argument(
            "--n_estimators",
            type=int,
            help=f"Random forest n_estimators. Default: {self.n_estimators}",
        )
        parser.add_argument(
            "--max_depth",
            type=int,
            help=f"Random forest max_depth. Default: {self.max_depth}",
        )
        parser.add_argument(
            "--min_samples_split",
            type=int,
            help=f"Random forest min_samples_split. Default: {self.min_samples_split}",
        )
        parser.add_argument(
            "--min_samples_leaf",
            type=int,
            help=f"Random forest min_samples_leaf. Default: {self.min_samples_leaf}",
        )

        # xgb hyperparams
        parser.add_argument(
            "--subsample", type=float, help=f"XGB subsample. Default: {self.subsample}"
        )
        parser.add_argument(
            "--colsample_bytree",
            type=float,
            help=f"XGB colsample_bytree. Default: {self.colsample_bytree}",
        )

        # training args
        parser.add_argument(
            "--save_model",
            type=str,
            choices=["true", "false"],
            help=f"Save model. Default: {self.save_model}",
        )
        parser.add_argument(
            "--save_predictions",
            type=str,
            choices=["true", "false"],
            help=f"Save predictions. Default: {self.save_predictions}",
        )
        parser.add_argument(
            "--verbose",
            type=str,
            choices=["true", "false"],
            help=f"Verbose output. Default: {self.verbose}",
        )

        # output args
        parser.add_argument(
            "--output_dir",
            type=str,
            help=f"Output directory. Default: {self.output_dir}",
        )
        parser.add_argument(
            "--experiment_name",
            type=str,
            help=f"Experiment name. Default: {self.experiment_name}",
        )
        parser.add_argument(
            "--seed", type=int, help=f"Random seed. Default: {self.seed}"
        )

        args = parser.parse_args()

        for k, v in vars(args).items():
            if v is not None and hasattr(self, k):
                if isinstance(getattr(self, k), bool):
                    setattr(self, k, v.lower() == "true")
                else:
                    setattr(self, k, v)
