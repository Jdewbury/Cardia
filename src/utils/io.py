import json
from pathlib import Path

import yaml


def save_file(file_path: Path, data: dict) -> None:
    """Save data to JSON or YAML filepath.

    Args:
        file_path: Path to save data
        data: Data to save
    """
    with open(file_path, "w") as f:
        if file_path.suffix == ".json":
            json.dump(data, f, indent=4)
        elif file_path.suffix in [".yaml", ".yml"]:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")


def load_file(file_path: Path) -> dict:
    """Load JSON or YAML file from filepath.

    Args:
        file_path: Path to desired file

    Returns:
        dict: Object of loaded file
    """
    with open(file_path, "r") as f:
        if file_path.suffix == ".json":
            data = json.load(f)
        elif file_path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return data


def make_dir(dir: Path | str) -> Path:
    """Make directory.

    Args:
        dir: Desired directory to initialize

    Returns:
        Initialized directory path
    """
    if isinstance(dir, str):
        dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)

    return dir
