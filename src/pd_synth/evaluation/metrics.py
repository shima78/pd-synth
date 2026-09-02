"""Synthetic-data quality metrics.

Lightweight, dependency-free (no scipy) statistics for comparing a batch of
real images against a batch of generated ones.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_image_difference(real: NDArray, synthetic: NDArray) -> float:
    """Mean absolute difference between the average real and average synthetic image.

    A cheap first sanity check: if the generator has collapsed or diverged,
    the aggregate appearance of its samples will differ sharply from real
    data and this number will be large.

    Args:
        real: ``(N, ...)`` array of real samples.
        synthetic: ``(M, ...)`` array of synthetic samples with matching
            trailing dimensions.

    Returns:
        Mean absolute difference between the two samples' mean images.
    """
    return float(np.abs(real.mean(axis=0) - synthetic.mean(axis=0)).mean())


def frechet_distance_diagonal(real_features: NDArray, synthetic_features: NDArray) -> float:
    """Diagonal-covariance Frechet (2-Wasserstein) distance between two feature sets.

    Operates on whatever feature representation is passed in - raw
    flattened pixels for a quick pipeline-validation signal, or classifier
    embeddings for a more meaningful one. Assumes independent feature
    dimensions (diagonal covariance) rather than computing a full matrix
    square root, so it needs no scipy dependency; treat it as a lightweight
    stand-in for a proper FID, not a replacement for one.

    Args:
        real_features: ``(N, D)`` array of real feature vectors.
        synthetic_features: ``(M, D)`` array of synthetic feature vectors.

    Returns:
        Non-negative scalar distance; ``0.0`` for identical distributions.
    """
    mu1, mu2 = real_features.mean(axis=0), synthetic_features.mean(axis=0)
    var1, var2 = real_features.var(axis=0), synthetic_features.var(axis=0)
    mean_term = np.sum((mu1 - mu2) ** 2)
    var_term = np.sum(var1 + var2 - 2.0 * np.sqrt(var1 * var2))
    return float(mean_term + var_term)
