"""Single source of randomness for the whole project.

Every script imports ``SEED`` from here and calls :func:`seed_everything` before
doing anything stochastic, so data splits, weight initialisation, and shuffling
are all reproducible from one constant. PyTorch is seeded only if it happens to
be installed (it is an M7 / benchmark-only dependency), so importing this module
never requires torch.
"""
from __future__ import annotations

import os
import random

SEED = 1337


def seed_everything(seed: int = SEED) -> int:
    """Seed Python, NumPy, and (if available) PyTorch from a single constant.

    Returns the seed so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    return seed
