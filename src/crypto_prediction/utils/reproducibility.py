"""Reproducibility utilities ensuring deterministic random seeds across all engines."""

from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, OS, and PyTorch (if available).

    Args:
        seed: The integer seed for deterministic execution.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional PyTorch determinism
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
