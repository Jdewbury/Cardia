import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set all seeds to make results reproducible.

    Args:
        seed: Desired seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
