"""Unit tests for pd_synth.sampling.boundary - the core thesis contribution.

Kept independent of any real generator/classifier: candidates and
predictions are plain tensors and callables, so these tests run instantly.
"""

from __future__ import annotations

import pytest
import torch

from pd_synth.sampling.boundary import (
    BoundaryFocusedSampler,
    boundary_score,
    select_boundary_samples,
)


def test_boundary_score_prefers_uncertain_predictions() -> None:
    probs = torch.tensor([[0.9, 0.1], [0.51, 0.49]])
    scores = boundary_score(probs)
    assert scores[1] < scores[0]


def test_boundary_score_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        boundary_score(torch.rand(4))


def test_select_boundary_samples_keeps_lowest_margin() -> None:
    images = torch.arange(3 * 1 * 2 * 2).reshape(3, 1, 2, 2).float()
    probs = torch.tensor([[0.9, 0.1], [0.51, 0.49], [0.6, 0.4]])

    selected, scores = select_boundary_samples(images, probs, num_samples=1)

    assert selected.shape == (1, 1, 2, 2)
    assert torch.equal(selected[0], images[1])
    assert scores.shape == (1,)


def test_select_boundary_samples_mismatched_lengths_raises() -> None:
    images = torch.rand(3, 1, 2, 2)
    probs = torch.rand(2, 2)
    with pytest.raises(ValueError):
        select_boundary_samples(images, probs, num_samples=1)


def test_boundary_focused_sampler_oversamples_then_filters() -> None:
    def generate_fn(n: int) -> torch.Tensor:
        return torch.arange(n * 1 * 2 * 2).reshape(n, 1, 2, 2).float()

    def classify_fn(images: torch.Tensor) -> torch.Tensor:
        n = images.shape[0]
        probs = torch.full((n, 2), 0.9)
        probs[:, 1] = 0.1
        probs[-1] = torch.tensor([0.5, 0.5])  # last candidate is most ambiguous
        return probs

    sampler = BoundaryFocusedSampler(generate_fn, classify_fn, oversample_factor=4.0)
    selected, scores = sampler.sample(num_samples=1)

    assert selected.shape == (1, 1, 2, 2)
    assert scores[0] < 0.9


def test_boundary_focused_sampler_rejects_invalid_oversample_factor() -> None:
    def generate_fn(n: int) -> torch.Tensor:
        return torch.rand(n, 1, 2, 2)

    def classify_fn(images: torch.Tensor) -> torch.Tensor:
        return torch.rand(images.shape[0], 2)

    with pytest.raises(ValueError):
        BoundaryFocusedSampler(generate_fn, classify_fn, oversample_factor=0.5)
