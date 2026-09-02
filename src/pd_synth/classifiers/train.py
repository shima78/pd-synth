"""Dataset-agnostic classifier train/eval loop.

Works with any ``torch.utils.data.DataLoader`` yielding ``(images, labels)``
batches, so the same code trains on real MNIST, synthetic MNIST, or (later)
real/synthetic gait-pose data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    """Outcome of a :func:`train_classifier` run."""

    train_losses: list[float]
    val_accuracy: float | None = None


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    num_epochs: int,
    lr: float,
    device: str = "cpu",
) -> TrainResult:
    """Train ``model`` with cross-entropy loss and Adam.

    Args:
        model: A classifier mapping images to class logits.
        train_loader: Yields ``(images, labels)`` training batches.
        val_loader: Optional loader to compute a final validation accuracy on.
        num_epochs: Number of passes over ``train_loader``.
        lr: Adam learning rate.
        device: torch device string.

    Returns:
        A :class:`TrainResult` with per-epoch training losses and, if
        ``val_loader`` was given, the final validation accuracy.
    """
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    for _ in range(num_epochs):
        model.train()
        running_loss, num_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device_t), labels.to(device_t)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        train_losses.append(running_loss / max(num_batches, 1))

    val_accuracy = None
    if val_loader is not None:
        val_accuracy = evaluate_classifier(model, val_loader, device)
    return TrainResult(train_losses=train_losses, val_accuracy=val_accuracy)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, data_loader: DataLoader, device: str = "cpu") -> float:
    """Compute classification accuracy of ``model`` over ``data_loader``.

    Args:
        model: A trained classifier mapping images to class logits.
        data_loader: Yields ``(images, labels)`` batches.
        device: torch device string.

    Returns:
        Fraction of correctly classified samples, in ``[0, 1]``. Returns
        ``0.0`` if ``data_loader`` is empty.
    """
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()
    correct, total = 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device_t), labels.to(device_t)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.numel()
    return correct / total if total else 0.0
