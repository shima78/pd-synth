"""Helpers for building class-balanced subsets of any LabeledImageDataset.

Used to make fair, data-budget-matched comparisons - e.g. a classifier
trained on N real images vs. one trained on N synthetic images.
"""

from __future__ import annotations

import random

from pd_synth.data.base import LabeledImageDataset


def index_by_class(dataset: LabeledImageDataset) -> dict[int, list[int]]:
    """Map each class label to the list of dataset indices carrying it.

    Args:
        dataset: Any dataset implementing the common (image, label) interface.

    Returns:
        A dict from class label to the indices in ``dataset`` with that label.
    """
    indices: dict[int, list[int]] = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        indices.setdefault(label, []).append(i)
    return indices


def balanced_subset_indices(
    indices_by_class: dict[int, list[int]], num_samples: int, num_classes: int, seed: int
) -> list[int]:
    """Pick a class-balanced random subset of indices.

    Splits ``num_samples`` as evenly as possible across ``num_classes`` and
    draws without replacement from each class's available indices.

    Args:
        indices_by_class: Output of :func:`index_by_class`.
        num_samples: Total number of indices to select.
        num_classes: Number of classes to split ``num_samples`` across.
        seed: Random seed, for a reproducible subset.

    Returns:
        A shuffled list of ``num_samples`` dataset indices.

    Raises:
        ValueError: If any class has fewer available indices than its share.
    """
    rng = random.Random(seed)
    base, remainder = divmod(num_samples, num_classes)
    selected: list[int] = []
    for cls in range(num_classes):
        want = base + (1 if cls < remainder else 0)
        available = indices_by_class.get(cls, [])
        if len(available) < want:
            raise ValueError(f"class {cls} has only {len(available)} samples, need {want}")
        selected.extend(rng.sample(available, want))
    rng.shuffle(selected)
    return selected
