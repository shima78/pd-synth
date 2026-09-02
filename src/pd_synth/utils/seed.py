"""Reproducibility helpers: a single seeding entry point for every run."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) RNGs for a reproducible run.

    Every experiment entry-point in ``experiments/`` should call this once,
    with the seed coming from the run's config file, before building any
    dataset, model, or dataloader.

    Args:
        seed: Integer seed applied to all RNGs used by this project.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
