"""MNIST behind the pd-synth dataset interface.

MNIST is used only to validate that the training/generation/sampling
pipeline works end-to-end. It is not a scarce dataset, so strong results
here are not evidence for the boundary-focused sampling hypothesis itself -
that requires the real, scarce gait/pose data this thesis targets. See the
top-level README for how to swap in that dataset later.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST

from pd_synth.data.base import LabeledImageDataset


class MNISTDataset(LabeledImageDataset):
    """Wraps ``torchvision.datasets.MNIST`` behind the common (image, label) interface.

    Images are scaled to ``[-1, 1]``, the range expected by the DDPM
    generator in :mod:`pd_synth.generation.diffusion`.
    """

    num_classes = 10

    def __init__(self, root: str | Path, train: bool = True, download: bool = True) -> None:
        """Load MNIST from (or download it to) ``root``.

        Args:
            root: Directory MNIST is stored in/downloaded to. Comes from
                config (e.g. ``data.root``) - never hardcoded.
            train: Whether to load the training split (``True``) or the
                test split (``False``).
            download: If ``True``, download MNIST to ``root`` when it is not
                already present there. Set to ``False`` to require the data
                to already exist locally (e.g. in offline environments).
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self._dataset = MNIST(root=str(root), train=train, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self._dataset[index]
        return image, int(label)
