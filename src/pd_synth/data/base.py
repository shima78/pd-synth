"""Common dataset interface shared by every dataset in this project.

Generation, sampling, and classifier code is written against
:class:`LabeledImageDataset` only - it never needs to know whether the
underlying data is MNIST (used to validate the pipeline mechanics) or the
real gait/pose data the thesis is ultimately about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import Dataset


class LabeledImageDataset(Dataset, ABC):
    """Abstract base class for all pd-synth datasets.

    Every implementation returns ``(image, label)`` pairs where ``image`` is
    a ``(C, H, W)`` float tensor and ``label`` is an integer class index.
    Subclasses must also set the ``num_classes`` class attribute.
    """

    num_classes: int

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return the ``(image, label)`` pair at ``index``."""
