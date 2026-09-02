"""Pure image-grid helper shared by experiment scripts and the demo app.

Kept dependency-free of any UI framework so it stays trivially unit-testable.
"""

from __future__ import annotations

from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def to_grid_image(images: Tensor, nrow: int) -> Image:
    """Rescale a ``[-1, 1]``-range image batch to ``[0, 1]`` and lay it out as a grid.

    Args:
        images: ``(N, C, H, W)`` batch of images scaled to ``[-1, 1]`` (the
            range :class:`~pd_synth.generation.DiffusionGenerator` trains and
            samples in).
        nrow: Number of images per row in the resulting grid.

    Returns:
        A single PIL image showing all of ``images`` arranged in a grid.
    """
    viewable = (images.detach().cpu().clamp(-1, 1) + 1) / 2
    return to_pil_image(make_grid(viewable, nrow=nrow))
