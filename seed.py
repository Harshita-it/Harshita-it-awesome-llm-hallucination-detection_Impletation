"""Random-seed control. Seeds python, numpy and (if installed) torch/CUDA."""
import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> dict:
    """Seed all random number generators we use.

    deterministic=True additionally asks PyTorch for deterministic algorithms
    (can be slower and may raise for some operations; we use warn_only=True).
    Returns a small dict describing what was seeded, for the log.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    info = {"seed": seed, "torch": False, "cuda": False, "deterministic": deterministic}
    try:
        import torch

        torch.manual_seed(seed)
        info["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            info["cuda"] = True
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return info


def seed_for_question(base: int, index: int) -> int:
    """Per-question sampling seed from the protocol: base (1000) + question index in the pool."""
    return int(base) + int(index)
