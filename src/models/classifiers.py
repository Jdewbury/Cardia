from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def initialize_random_forest(
    params: dict, random_state: int = 42
) -> RandomForestClassifier:
    """Create Random Forest classifier with set parameters.

    Args:
        params: Dictionary containing model hyperparameters
        random_state: Random seed

    Returns:
        RandomForestClassifier instance (not fitted)
    """
    return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)


def initialize_xgboost(params: dict, random_state: int = 42) -> XGBClassifier:
    """Create XGBoost classifier with set parameters.

    Args:
        params: Dictionary containing model hyperparameters
        random_state: Random seed

    Returns:
        XGBClassifier instance (not fitted)
    """
    return XGBClassifier(
        **params, random_state=random_state, n_jobs=-1, eval_metric="mlogloss"
    )
