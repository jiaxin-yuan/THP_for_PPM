"""
utils/reproducibility.py
------------------------
Seed-setting utilities for deterministic experiments.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> torch.Generator:
    """Fix all random-number generators for full reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, PyTorch (CPU + all
    CUDA devices), and disables cuDNN non-determinism.

    Args:
        seed:  Integer seed value.

    Returns:
        A :class:`torch.Generator` already seeded with *seed*, ready to be
        passed to DataLoader or other PyTorch samplers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)