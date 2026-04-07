import numpy as np
import pandas as pd

# PAMAP2 dataset constants
PAMAP2_ACTIVITY_MAPPING = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "stationary", # standing
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "driving_car",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}
PAMAP2_SAMPLING_RATE = 100

PAMAP2_INTENSITY_GROUPS = {
    "light": [1, 2, 3, 9, 10, 11, 17, 18, 19],  # lying, sitting, standing, watching_tv, computer_work, driving_car, ironing, folding_laundry, house_cleaning
    "moderate": [4, 6, 7, 13, 16],              # walking, cycling, nordic_walking, descending_stairs, vacuum_cleaning
    "vigorous": [5, 12, 20, 24],                # running, ascending_stairs, playing_soccer, rope_jumping
}
PAMAP2_INTENSITY_GROUP_MAPPING = {
    id: group for group, ids in PAMAP2_INTENSITY_GROUPS.items() for id in ids
}

PAMAP2_INTENSITY_ID_MAPPING = {
    1: "light",
    2: "moderate",
    3: "vigorous",
}

def get_class_labels(activity_ids: np.ndarray, intensity_groups: bool = False) -> np.ndarray:
    """Convert activity array of ID to names.
    
    Args:
        activity_ids: Array of activity IDs
        intensity_groups: Boolean to group activities together by intensity
        
    Returns:
        Array of activity name strings
    """
    mapping = PAMAP2_INTENSITY_ID_MAPPING if intensity_groups else PAMAP2_ACTIVITY_MAPPING
    
    return np.array([
        mapping.get(id, f"Unknown_{id}") for id in activity_ids
    ])
    
def map_to_intensity_groups(activity_ids: np.ndarray) -> np.ndarray:
    """Convert activity IDs to intensity group IDs.
    
    Args:
        activity_ids: Array of activity IDs
        
    Returns:
        Array of intensity group IDs
    """
    intensity_name_to_id = {
        "light": 1,
        "moderate": 2,
        "vigorous": 3,
    }
    
    activity_to_intensity_id = {
        activity_id: intensity_name_to_id[group_name]
        for activity_id, group_name in PAMAP2_INTENSITY_GROUP_MAPPING.items()
    }
    
    return np.array([
        activity_to_intensity_id.get(id, f"Unknown_{id}") for id in activity_ids
    ])
    
# granular label remapping
PAMAP2_ACTIVITY_REMAPPING = {
    17: 3,   # ironing -> standing
    2: 3,    # sitting -> standing
    16: 4,   # vacuum_cleaning -> walking
    7: 4,    # nordic_walking -> walking
    18: 19,  # folding_laundry -> house_cleaning
    10: 9,   # computer_work -> watching_tv
}

def remap_similar_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Combine similar activities into single labels.
    
    Args:
        df: DataFrame with activity_id column
        
    Returns:
        DataFrame with remapped activity IDs
    """
    df["activity_id"] = df["activity_id"].replace(PAMAP2_ACTIVITY_REMAPPING)
    
    return df