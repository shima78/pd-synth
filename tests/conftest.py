"""Shared pytest fixtures.

Fixtures here generate tiny synthetic (image, label) data and write it to
disk, so the whole test suite runs offline - no network access and no
dependency on the real MNIST download.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from pd_synth.data.base import LabeledImageDataset


class TinyImageDataset(LabeledImageDataset):
    """A tiny in-memory dataset used only in tests, mirroring the real interface."""

    num_classes = 10

    def __init__(self, images: Tensor, labels: Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have matching length")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.images[index], int(self.labels[index])


@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> Path:
    """Write a tiny synthetic (image, label) dataset to disk and return its path."""
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(8, 1, 8, 8, generator=generator) * 2 - 1  # roughly [-1, 1]
    labels = torch.randint(0, 10, (8,), generator=generator)
    data_path = tmp_path / "tiny_dataset.pt"
    torch.save({"images": images, "labels": labels}, data_path)
    return data_path


@pytest.fixture
def tiny_dataset(tiny_dataset_path: Path) -> TinyImageDataset:
    """Load the on-disk tiny dataset fixture into a :class:`TinyImageDataset`."""
    payload = torch.load(tiny_dataset_path)
    return TinyImageDataset(payload["images"], payload["labels"])
