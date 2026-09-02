"""Classifier evaluation metrics: confusion matrix, precision, recall, F1.

Implemented directly on torch tensors (no scikit-learn dependency) so they
work for any number of classes on either MNIST or the real dataset later.
"""

from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(preds: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    """Build a ``(num_classes, num_classes)`` confusion matrix.

    Args:
        preds: Predicted class indices, any shape.
        labels: True class indices, same shape as ``preds``.
        num_classes: Total number of classes.

    Returns:
        Integer matrix where entry ``[i, j]`` counts samples with true label
        ``i`` predicted as class ``j``.
    """
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, pred_label in zip(labels.reshape(-1), preds.reshape(-1), strict=True):
        matrix[true_label.long(), pred_label.long()] += 1
    return matrix


def precision_recall_f1(preds: Tensor, labels: Tensor, num_classes: int) -> dict[str, Tensor]:
    """Compute per-class precision, recall, and F1 score.

    Args:
        preds: Predicted class indices, any shape.
        labels: True class indices, same shape as ``preds``.
        num_classes: Total number of classes.

    Returns:
        Dict with ``"precision"``, ``"recall"``, and ``"f1"``, each a
        ``(num_classes,)`` float tensor. Classes with no predictions (for
        precision) or no true instances (for recall) score ``0.0`` rather
        than dividing by zero.
    """
    cm = confusion_matrix(preds, labels, num_classes).float()
    true_positives = cm.diag()
    predicted_positives = cm.sum(dim=0)
    actual_positives = cm.sum(dim=1)

    zeros = torch.zeros_like(true_positives)
    precision = torch.where(predicted_positives > 0, true_positives / predicted_positives, zeros)
    recall = torch.where(actual_positives > 0, true_positives / actual_positives, zeros)
    f1_denominator = precision + recall
    f1 = torch.where(f1_denominator > 0, 2 * precision * recall / f1_denominator, zeros)
    return {"precision": precision, "recall": recall, "f1": f1}
