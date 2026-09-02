"""Unit tests for pd_synth.evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pd_synth.evaluation.classifier_metrics import confusion_matrix, precision_recall_f1
from pd_synth.evaluation.metrics import frechet_distance_diagonal, mean_image_difference


def test_frechet_distance_diagonal_zero_for_identical_features() -> None:
    features = np.random.default_rng(0).normal(size=(20, 4))
    assert frechet_distance_diagonal(features, features) == pytest.approx(0.0, abs=1e-8)


def test_frechet_distance_diagonal_positive_for_different_features() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(loc=0.0, size=(20, 4))
    synthetic = rng.normal(loc=5.0, size=(20, 4))
    assert frechet_distance_diagonal(real, synthetic) > 0.0


def test_mean_image_difference_zero_for_identical_batches() -> None:
    batch = np.random.default_rng(0).random((5, 3, 3))
    assert mean_image_difference(batch, batch) == pytest.approx(0.0)


def test_confusion_matrix_and_precision_recall_f1_known_values() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 1])

    cm = confusion_matrix(preds, labels, num_classes=2)
    assert torch.equal(cm, torch.tensor([[1, 1], [0, 2]]))

    metrics = precision_recall_f1(preds, labels, num_classes=2)
    assert metrics["recall"][0].item() == pytest.approx(0.5)
    assert metrics["recall"][1].item() == pytest.approx(1.0)
    assert metrics["precision"][1].item() == pytest.approx(2 / 3)


def test_precision_recall_f1_handles_unpredicted_class() -> None:
    # Class 0 is never predicted, so its precision must be 0.0, not NaN.
    labels = torch.tensor([0, 1])
    preds = torch.tensor([1, 1])

    metrics = precision_recall_f1(preds, labels, num_classes=2)

    assert metrics["precision"][0].item() == 0.0
    assert metrics["recall"][0].item() == 0.0
