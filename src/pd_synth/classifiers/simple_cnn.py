"""A small, dataset-agnostic CNN classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleCNN(nn.Module):
    """A small CNN sized entirely from config (channels, image size, classes).

    Used both to evaluate synthetic-data quality (train on synthetic, test
    on real) and as the classifier that boundary-focused sampling scores
    candidates against. The same architecture works for MNIST now and the
    real gait/pose image representations later - nothing here is
    MNIST-specific.
    """

    def __init__(self, in_channels: int, image_size: int, num_classes: int) -> None:
        """Build the network.

        Args:
            in_channels: Number of input image channels.
            image_size: Height/width of (square) input images. Must be
                divisible by 4, since the network downsamples twice.
            num_classes: Number of output classes.

        Raises:
            ValueError: If ``image_size`` is not divisible by 4.
        """
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError(f"image_size must be divisible by 4, got {image_size}")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled_size = image_size // 4
        self.classifier = nn.Linear(32 * pooled_size * pooled_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw class logits for a ``(N, C, H, W)`` batch of images."""
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
