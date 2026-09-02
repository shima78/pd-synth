"""Stage 1 entry point: train the classifier on the configured (real) dataset.

Usage:
    python experiments/train_classifier.py --config configs/mnist_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pd_synth.classifiers import SimpleCNN, train_classifier
from pd_synth.data import get_dataset
from pd_synth.utils import load_config, save_config, set_seed


def main(config_path: str) -> None:
    """Train a classifier per ``config_path`` and save the model, metrics, and config."""
    config = load_config(config_path)
    set_seed(config["seed"])

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "classifier_config.yaml")

    data_cfg = config["data"]
    train_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=True,
        download=data_cfg.get("download", True),
    )
    val_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=False,
        download=data_cfg.get("download", True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_cfg["batch_size"], num_workers=data_cfg.get("num_workers", 0)
    )

    clf_cfg = config["classifier"]
    model = SimpleCNN(
        in_channels=clf_cfg["in_channels"],
        image_size=clf_cfg["image_size"],
        num_classes=clf_cfg["num_classes"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=clf_cfg["num_epochs"],
        lr=clf_cfg["lr"],
        device=device,
    )
    print(f"Train losses: {result.train_losses}")
    print(f"Validation accuracy: {result.val_accuracy}")

    torch.save(model.state_dict(), output_dir / "classifier.pt")
    save_config(
        {"train_losses": result.train_losses, "val_accuracy": result.val_accuracy},
        output_dir / "classifier_metrics.yaml",
    )
    print(f"Saved classifier, metrics, and config to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    main(args.config)
