"""Smoke tests for pd_synth.data.

The MNIST test deliberately never sets download=True, so this file never
touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pd_synth.data import InMemoryDataset, balanced_subset_indices, get_dataset, index_by_class
from pd_synth.data.mnist import MNISTDataset


def test_tiny_dataset_matches_common_interface(tiny_dataset) -> None:
    image, label = tiny_dataset[0]
    assert image.shape == (1, 8, 8)
    assert isinstance(label, int)
    assert len(tiny_dataset) == 8


def test_mnist_without_local_files_fails_fast_offline(tmp_path: Path) -> None:
    # download=False must never hit the network; with no local files present
    # it should raise immediately instead of attempting a download.
    with pytest.raises(RuntimeError):
        MNISTDataset(root=tmp_path, train=True, download=False)


def test_get_dataset_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        get_dataset("not-a-real-dataset")


def test_in_memory_dataset_matches_common_interface() -> None:
    images = torch.rand(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])

    dataset = InMemoryDataset(images, labels, num_classes=10)

    assert len(dataset) == 4
    image, label = dataset[2]
    assert torch.equal(image, images[2])
    assert label == 2
    assert isinstance(label, int)


def test_in_memory_dataset_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        InMemoryDataset(torch.rand(4, 1, 8, 8), torch.tensor([0, 1]), num_classes=10)


def test_index_by_class_groups_indices_by_label() -> None:
    dataset = InMemoryDataset(torch.rand(5, 1, 4, 4), torch.tensor([0, 1, 0, 1, 0]), num_classes=2)

    grouped = index_by_class(dataset)

    assert grouped[0] == [0, 2, 4]
    assert grouped[1] == [1, 3]


def test_balanced_subset_indices_splits_evenly_and_is_reproducible() -> None:
    grouped = {0: [0, 1, 2, 3], 1: [10, 11, 12, 13]}

    subset_a = balanced_subset_indices(grouped, num_samples=4, num_classes=2, seed=0)
    subset_b = balanced_subset_indices(grouped, num_samples=4, num_classes=2, seed=0)

    assert sorted(subset_a) == sorted(subset_b)
    assert len(subset_a) == 4
    assert sum(1 for i in subset_a if i in grouped[0]) == 2
    assert sum(1 for i in subset_a if i in grouped[1]) == 2


def test_balanced_subset_indices_raises_when_class_too_small() -> None:
    grouped = {0: [0, 1], 1: [10]}
    with pytest.raises(ValueError):
        balanced_subset_indices(grouped, num_samples=4, num_classes=2, seed=0)
