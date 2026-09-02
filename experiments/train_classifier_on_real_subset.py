"""Stage 1 entry point: train a classifier on a size-matched real-data subset.

The plain real-data baseline (``train_classifier.py``) trains on the full
60,000-image MNIST training set, while the synthetic-data classifier
(``train_classifier_on_synthetic.py``) trains on only
``sampling.num_synthetic_samples`` generated images - so comparing their
accuracies conflates "real vs. synthetic" with "60x more data vs. less
data". This script trains on a class-balanced *real* subset of exactly the
same size as the synthetic training set, so all three numbers are directly
comparable on equal data budget.

Usage:
    python experiments/train_classifier_on_real_subset.py --config configs/mnist_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from pd_synth.classifiers import SimpleCNN, train_classifier
from pd_synth.data import balanced_subset_indices, get_dataset, index_by_class
from pd_synth.utils import load_config, save_config, set_seed


def main(config_path: str) -> None:
    """Train a classifier on a size-matched real-data subset per ``config_path``."""
    config = load_config(config_path)
    set_seed(config["seed"])

    output_dir = Path(config["output"]["dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf_cfg = config["classifier"]
    num_classes = clf_cfg["num_classes"]
    sampling_cfg = config["sampling"]
    num_train_samples = sampling_cfg["num_synthetic_samples"]

    data_cfg = config["data"]
    real_train_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=True,
        download=data_cfg.get("download", True),
    )
    indices = balanced_subset_indices(
        index_by_class(real_train_dataset), num_train_samples, num_classes, config["seed"]
    )
    subset_dataset = Subset(real_train_dataset, indices)
    subset_loader = DataLoader(subset_dataset, batch_size=data_cfg["batch_size"], shuffle=True)

    real_test_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=False,
        download=data_cfg.get("download", True),
    )
    real_test_loader = DataLoader(real_test_dataset, batch_size=data_cfg["batch_size"])

    model = SimpleCNN(
        in_channels=clf_cfg["in_channels"],
        image_size=clf_cfg["image_size"],
        num_classes=num_classes,
    )
    result = train_classifier(
        model,
        subset_loader,
        real_test_loader,
        num_epochs=clf_cfg["num_epochs"],
        lr=clf_cfg["lr"],
        device=device,
    )
    print(f"Train losses (on {num_train_samples} real images): {result.train_losses}")
    print(f"Validation accuracy (on real MNIST test set): {result.val_accuracy}")

    torch.save(model.state_dict(), output_dir / "classifier_on_real_subset.pt")
    save_config(config, output_dir / "classifier_on_real_subset_config.yaml")
    save_config(
        {
            "train_losses": result.train_losses,
            "val_accuracy": result.val_accuracy,
            "num_real_train_samples": num_train_samples,
        },
        output_dir / "classifier_on_real_subset_metrics.yaml",
    )
    print(f"Saved classifier, metrics, and config to {output_dir}")

    full_metrics_path = output_dir / "classifier_metrics.yaml"
    synthetic_metrics_path = output_dir / "classifier_on_synthetic_metrics.yaml"
    if full_metrics_path.exists() and synthetic_metrics_path.exists():
        full_accuracy = load_config(full_metrics_path)["val_accuracy"]
        synthetic_accuracy = load_config(synthetic_metrics_path)["val_accuracy"]
        print(
            f"Comparison (all on the real MNIST test set): "
            f"full real data (60,000) {full_accuracy:.4f} vs. "
            f"{num_train_samples} real images {result.val_accuracy:.4f} vs. "
            f"{num_train_samples} synthetic images {synthetic_accuracy:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    main(args.config)
