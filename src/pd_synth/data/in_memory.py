"""A dataset-agnostic wrapper for already-in-memory (image, label) tensors.

Used to feed generated synthetic samples - or any other tensor pair - through
the same :class:`LabeledImageDataset` interface as disk-backed datasets, so
downstream code (classifiers, evaluation) never has to special-case them.
"""

from __future__ import annotations

from torch import Tensor

from pd_synth.data.base import LabeledImageDataset


class InMemoryDataset(LabeledImageDataset):
    """Wraps a ``(images, labels)`` tensor pair behind the common dataset interface."""

    def __init__(self, images: Tensor, labels: Tensor, num_classes: int) -> None:
        """Store the tensors.

        Args:
            images: ``(N, C, H, W)`` image tensor.
            labels: ``(N,)`` integer class labels, aligned with ``images``.
            num_classes: Total number of classes labels are drawn from.

        Raises:
            ValueError: If ``images`` and ``labels`` have mismatched length.
        """
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.images[index], int(self.labels[index])
