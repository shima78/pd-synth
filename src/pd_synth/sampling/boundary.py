"""Boundary-focused sampling: the core thesis contribution.

The hypothesis this thesis tests is that synthetic samples a classifier
finds most ambiguous - i.e. those closest to a decision boundary between
classes - carry more training value than uniformly-sampled synthetic data.
This module is deliberately independent of any specific generator or
classifier implementation (it only depends on plain tensors and callables),
so it stays unit-testable in isolation and reusable once MNIST is swapped
for the real gait/pose data.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def boundary_score(probs: Tensor) -> Tensor:
    """Score each sample by closeness to a decision boundary.

    The score is the margin between a sample's top-1 and top-2 predicted
    class probabilities. A small margin means the classifier is nearly torn
    between two classes, i.e. the sample sits close to a decision boundary.

    Args:
        probs: ``(N, C)`` class probabilities (e.g. softmax output).

    Returns:
        ``(N,)`` tensor of margins; lower values are more boundary-like.

    Raises:
        ValueError: If ``probs`` is not 2-dimensional.
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs of shape (N, C), got {tuple(probs.shape)}")
    if probs.shape[1] < 2:
        return torch.zeros(probs.shape[0], device=probs.device)
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def select_boundary_samples(
    images: Tensor, probs: Tensor, num_samples: int
) -> tuple[Tensor, Tensor]:
    """Keep the ``num_samples`` images with the smallest boundary margin.

    Args:
        images: ``(N, C, H, W)`` candidate images.
        probs: ``(N, C)`` class probabilities aligned with ``images``.
        num_samples: How many of the most boundary-like images to keep.
            Clamped to ``N`` if larger.

    Returns:
        A tuple of ``(selected_images, selected_scores)``, both sorted from
        most to least boundary-like.

    Raises:
        ValueError: If ``images`` and ``probs`` have mismatched batch sizes.
    """
    if images.shape[0] != probs.shape[0]:
        raise ValueError("images and probs must have the same number of rows")
    num_samples = min(num_samples, images.shape[0])
    scores = boundary_score(probs)
    order = torch.argsort(scores)[:num_samples]
    return images[order], scores[order]


class BoundaryFocusedSampler:
    """Oversample-and-filter strategy: generate candidates, keep the most boundary-like.

    Wraps a generator's sampling function and a classifier's scoring
    function behind a single ``sample`` call. Both are injected as plain
    callables so this class has no dependency on
    :mod:`pd_synth.generation` or :mod:`pd_synth.classifiers`, and can be
    tested with trivial stand-ins.
    """

    def __init__(
        self,
        generate_fn: Callable[[int], Tensor],
        classify_fn: Callable[[Tensor], Tensor],
        oversample_factor: float = 4.0,
    ) -> None:
        """Configure the sampler.

        Args:
            generate_fn: Given a candidate count, returns a
                ``(count, C, H, W)`` tensor of generated images.
            classify_fn: Given a batch of images, returns ``(count, num_classes)``
                class probabilities.
            oversample_factor: How many extra candidates to generate per
                requested sample before filtering down (must be >= 1.0).

        Raises:
            ValueError: If ``oversample_factor`` is less than 1.0.
        """
        if oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1.0")
        self.generate_fn = generate_fn
        self.classify_fn = classify_fn
        self.oversample_factor = oversample_factor

    def sample(self, num_samples: int) -> tuple[Tensor, Tensor]:
        """Generate candidates and return the ``num_samples`` most boundary-like.

        Args:
            num_samples: Number of synthetic samples to return.

        Returns:
            A tuple of ``(selected_images, selected_scores)``.
        """
        num_candidates = max(num_samples, int(num_samples * self.oversample_factor))
        candidates = self.generate_fn(num_candidates)
        probs = self.classify_fn(candidates)
        return select_boundary_samples(candidates, probs, num_samples)
