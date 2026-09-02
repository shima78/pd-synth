"""Smoke test for pd_synth.utils.viz."""

from __future__ import annotations

import torch
from PIL.Image import Image

from pd_synth.utils.viz import to_grid_image


def test_to_grid_image_returns_pil_image_of_expected_size() -> None:
    images = torch.rand(4, 1, 8, 8) * 2 - 1  # in [-1, 1], like the generator's output

    grid = to_grid_image(images, nrow=2)

    assert isinstance(grid, Image)
    # 2x2 grid of 8x8 images, plus torchvision's default 2px padding around each.
    assert grid.size == (22, 22)
