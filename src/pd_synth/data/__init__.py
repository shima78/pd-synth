"""Dataset registry: look up a :class:`LabeledImageDataset` implementation by name.

Adding a new dataset (e.g. the real gait/pose data) means writing a class
that implements :class:`LabeledImageDataset` and registering it in
``_REGISTRY`` below - nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from typing import Any

from pd_synth.data.base import LabeledImageDataset
from pd_synth.data.in_memory import InMemoryDataset
from pd_synth.data.mnist import MNISTDataset
from pd_synth.data.subset import balanced_subset_indices, index_by_class

_REGISTRY: dict[str, type[LabeledImageDataset]] = {
    "mnist": MNISTDataset,
}


def get_dataset(name: str, **kwargs: Any) -> LabeledImageDataset:
    """Instantiate a registered dataset by name.

    Args:
        name: Registry key, e.g. ``"mnist"`` (case-insensitive).
        **kwargs: Forwarded to the dataset class's constructor.

    Returns:
        An instantiated :class:`LabeledImageDataset`.

    Raises:
        ValueError: If ``name`` is not a registered dataset.
    """
    try:
        dataset_cls = _REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}") from exc
    return dataset_cls(**kwargs)


__all__ = [
    "InMemoryDataset",
    "LabeledImageDataset",
    "MNISTDataset",
    "balanced_subset_indices",
    "get_dataset",
    "index_by_class",
]
