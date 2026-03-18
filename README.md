# Cardia

A modular machine learning framework for human activity recognition using wearable sensor data integrated into a Holter monitor.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)

## Overview

This project implements classical machine learning models (Random Forest and XGBoost) for activity recognition. The framework supports:

- **Multiple ML models**: Random Forest and XGBoost classifiers
- **Comprehensive hyperparameter sweeps**: Window size, sampling rate, and model-specific hyperparameters
- **Flexible data preprocessing**: Configurable windowing, feature extraction, and activity grouping

### Current Features

- Activity classification from PAMAP2 chest IMU data

## Project Structure

```
cardia/
├── scripts/                      # Experiment entry points
│   ├── train_model.py            # Train and evaluate a single model
│   ├── hyperparameter_sweep.py   # Grid search for model hyperparameters
│   ├── window_sweep.py           # Find optimal window size
│   └── sampling_rate_sweep.py    # Find optimal sampling rate
│
├── src/
│   ├── config.py                 # Configuration management
│   │
│   ├── data/                     # Data loading and preprocessing
│   │   ├── constants.py          # Dataset-specific mappings and constants
│   │   ├── dataset.py            # Windowing and data splitting utilities
│   │   ├── features.py           # Feature extraction functions
│   │   └── pamap2.py             # PAMAP2 dataset loader
│   │
│   ├── models/                   # Model implementations
│   │   └── classifiers.py        # ML model initialization
│   │
│   ├── training/                 # Training and sweeping logic
│   │   ├── train_classical.py    # Classical ML training pipeline
│   │   └── sweep.py              # Hyperparameter sweep implementations
│   │
│   ├── evaluation/               # Metrics and visualization
│   │   └── metrics.py            # Evaluation functions and plotting
│   │
│   └── utils/                    # General utilities
│       ├── io.py                 # File I/O helpers
│       └── reproducibility.py    # Seed management
│
├── experiments/                  # Output directory for results
└── README.md                     # This file
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cardia.git
cd cardia
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages include:
- numpy
- pandas
- scikit-learn
- xgboost
- matplotlib
- tqdm
- pyyaml
- torch

### Chest IMU Data Setup

1. Download the PAMAP2 dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

2. Extract the dataset and note the path to the `Protocol` directory

3. Update the `data_dir` in your configuration or pass it as a command-line argument

## Configuration
You can modify the default arguments in the [config](src\config.py) file, or specify CLI arguments when running individual scripts.

Full CLI:

```
usage: train_model.py [-h] [--data_dir DATA_DIR] [--dataset DATASET] [--filter_chest {true,false}] [--exclude_sensors EXCLUDE_SENSORS [EXCLUDE_SENSORS ...]]    
                      [--group_activities {true,false}] [--combine_similar {true,false}] [--use_all_data {true,false}] [--use_heart_rate {true,false}]
                      [--window_size_sec WINDOW_SIZE_SEC] [--overlap OVERLAP] [--sampling_rate SAMPLING_RATE] [--model_name {random_forest,xgboost}]
                      [--learning_rate LEARNING_RATE] [--n_estimators N_ESTIMATORS] [--max_depth MAX_DEPTH] [--min_samples_split MIN_SAMPLES_SPLIT]
                      [--min_samples_leaf MIN_SAMPLES_LEAF] [--subsample SUBSAMPLE] [--colsample_bytree COLSAMPLE_BYTREE] [--save_model {true,false}]
                      [--save_predictions {true,false}] [--verbose {true,false}] [--output_dir OUTPUT_DIR] [--experiment_name EXPERIMENT_NAME] [--seed SEED]    

Cardia Experimental Config

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to data directory. Default: data\pamap2+physical+activity+monitoring\PAMAP2_Dataset\PAMAP2_Dataset\Protocol
  --dataset DATASET     Dataset name. Default: pamap2
  --filter_chest {true,false}
                        Filter chest sensors. Default: True
  --exclude_sensors EXCLUDE_SENSORS [EXCLUDE_SENSORS ...]
                        Sensor types to exclude (e.g., mag_x acc_x_2 acc_y_2 acc_z_2)
  --group_activities {true,false}
                        Group activities by intensity. Default: False
  --combine_similar {true,false}
                        Group similar granular activities. Default: True
  --use_all_data {true,false}
                        Include all data into training. Default: False
  --use_heart_rate {true,false}
                        Include heart rate in features. Default: False
  --window_size_sec WINDOW_SIZE_SEC
                        Window size in seconds. Default: 5.0
  --overlap OVERLAP     Window overlap fraction. Default: 0.5
  --sampling_rate SAMPLING_RATE
                        Sampling rate. Default: 100
  --model_name {random_forest,xgboost}
                        Model name. Default: random_forest
  --learning_rate LEARNING_RATE
                        Learning rate. Default: 0.1
  --n_estimators N_ESTIMATORS
                        Random forest n_estimators. Default: 50
  --max_depth MAX_DEPTH
                        Random forest max_depth. Default: 25
  --min_samples_split MIN_SAMPLES_SPLIT
                        Random forest min_samples_split. Default: 2
  --min_samples_leaf MIN_SAMPLES_LEAF
                        Random forest min_samples_leaf. Default: 1
  --subsample SUBSAMPLE
                        XGB subsample. Default: 0.7
  --colsample_bytree COLSAMPLE_BYTREE
                        XGB colsample_bytree. Default: 0.8
  --save_model {true,false}
                        Save model. Default: True
  --save_predictions {true,false}
                        Save predictions. Default: False
  --verbose {true,false}
                        Verbose output. Default: True
  --output_dir OUTPUT_DIR
                        Output directory. Default: models
  --experiment_name EXPERIMENT_NAME
                        Experiment name. Default: experiments
  --seed SEED           Random seed. Default: 42
```

## Usage

### Training a Model

Train a Random Forest classifier with default parameters:

```bash
python scripts/train_model.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name random_forest \
    --window_size_sec 5.0 \
    --experiment_name rf_experiment
```

Train an XGBoost classifier:

```bash
python scripts/train_model.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name xgboost \
    --window_size_sec 5.0 \
    --experiment_name xgb_experiment
```

### Hyperparameter Sweeps

#### Model Hyperparameters

Find the best Random Forest hyperparameters:

```bash
python scripts/hyperparameter_sweep.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name random_forest \
    --experiment_name rf_hyperparam_sweep
```

Find the best XGBoost hyperparameters:

```bash
python scripts/hyperparameter_sweep.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name xgboost \
    --experiment_name xgb_hyperparam_sweep
```

#### Window Size Sweep

Test different window lengths (1-30 seconds):

```bash
python scripts/window_sweep.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name random_forest \
    --experiment_name window_sweep
```

#### Sampling Rate Sweep

Test different sampling rates:

```bash
python scripts/sampling_rate_sweep.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name random_forest \
    --experiment_name sampling_sweep
```

### Activity Grouping

Group activities by intensity level (light, moderate, vigorous):

```bash
python scripts/train_model.py \
    --data_dir /path/to/PAMAP2_Dataset/ \
    --model_name random_forest \
    --group_activities true \
    --experiment_name intensity_classification
```

### Output Files

Each experiment creates a directory in `experiments/` containing:

- `config.json` - Complete configuration used for the experiment
- `results.json` - Performance metrics
- `cm_val.png`, `cm_test.png` - Confusion matrices
- `predictions_val.npz`, `predictions_test.npz` - Model predictions
- `confidence_analysis.csv`, `confidence_analysis.png`, `confidence_summary.json` - Prediction confidence breakdown by class
- `feature_importance.csv`, `feature_importance.png` - Feature importance (Random Forest only)
- `*_model.pkl` - Trained model (if `--save_model true`)
- `label_encoder.pkl` - Label encoder (XGBoost only)

Sweep scripts produce additional outputs:
- `rf_sweep_results.csv`, `rf_best_params.json` - Random Forest hyperparameter sweep
- `xgb_sweep_results.csv`, `xgb_best_params.json` - XGBoost hyperparameter sweep
- `*_window_sweep_results.csv`, `*_window_best_params.json` - Window size sweep
- `*_sampling_rate_results.csv`, `*_sampling_rate_best_params.json` - Sampling rate sweep