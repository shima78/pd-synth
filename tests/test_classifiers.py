"""Smoke tests for pd_synth.classifiers."""

from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from pd_synth.classifiers.simple_cnn import SimpleCNN
from pd_synth.classifiers.train import evaluate_classifier, train_classifier


def test_simple_cnn_rejects_bad_image_size() -> None:
    with pytest.raises(ValueError):
        SimpleCNN(in_channels=1, image_size=7, num_classes=10)


def test_train_and_evaluate_classifier(tiny_dataset) -> None:
    model = SimpleCNN(in_channels=1, image_size=8, num_classes=10)
    loader = DataLoader(tiny_dataset, batch_size=4)

    result = train_classifier(model, loader, loader, num_epochs=1, lr=1e-3, device="cpu")

    assert len(result.train_losses) == 1
    assert result.val_accuracy is not None
    assert 0.0 <= result.val_accuracy <= 1.0

    accuracy = evaluate_classifier(model, loader, device="cpu")
    assert 0.0 <= accuracy <= 1.0
