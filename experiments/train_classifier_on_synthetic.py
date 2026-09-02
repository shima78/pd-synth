"""Stage 1 entry point: train a classifier on synthetic data, evaluate on real data.

Loads the class-conditional generator trained by ``train_generator.py``,
generates a class-balanced labeled synthetic training set (true labels, not
pseudo-labels, since the generator is conditioned on the real class), trains
a classifier on it, and evaluates on the same real MNIST test set the
real-data classifier (``train_classifier.py``) was evaluated on - so the two
val_accuracy numbers are directly comparable.

Usage:
    python experiments/train_classifier_on_synthetic.py --config configs/mnist_baseline.yaml

Pass ``--label NAME`` to tag the outputs (``classifier_on_synthetic_NAME*``)
so a run at a different ``sampling.num_synthetic_samples`` sits alongside the
default one instead of overwriting it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pd_synth.classifiers import SimpleCNN, train_classifier
from pd_synth.data import InMemoryDataset, get_dataset
from pd_synth.generation import DiffusionConfig, DiffusionGenerator
from pd_synth.utils import load_config, save_config, set_seed


def main(config_path: str, label: str | None = None) -> None:
    """Train a classifier on generator-synthesized data per ``config_path``.

    Args:
        config_path: Path to a YAML experiment config.
        label: Optional tag appended to the output filenames
            (``classifier_on_synthetic_<label>*``), so runs at different
            ``sampling.num_synthetic_samples`` do not overwrite each other.
    """
    config = load_config(config_path)
    set_seed(config["seed"])

    suffix = f"_{label}" if label else ""
    output_dir = Path(config["output"]["dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_cfg = config["generator"]
    generator = DiffusionGenerator(
        DiffusionConfig(
            image_size=gen_cfg["image_size"],
            in_channels=gen_cfg["in_channels"],
            out_channels=gen_cfg["out_channels"],
            layers_per_block=gen_cfg["layers_per_block"],
            block_out_channels=tuple(gen_cfg["block_out_channels"]),
            num_train_timesteps=gen_cfg["num_train_timesteps"],
            norm_num_groups=gen_cfg["norm_num_groups"],
        ),
        device=device,
    )
    generator.load_pretrained(output_dir / "generator")
    if not generator.class_conditional:
        raise RuntimeError(
            "Generator at "
            f"{output_dir / 'generator'} is not class-conditional "
            "(train with generator.class_conditional: true first)."
        )

    clf_cfg = config["classifier"]
    num_classes = clf_cfg["num_classes"]
    sampling_cfg = config["sampling"]
    num_synthetic_samples = sampling_cfg["num_synthetic_samples"]

    # Class-balanced labels: as close to an even split across classes as
    # num_synthetic_samples allows.
    labels = torch.arange(num_synthetic_samples) % num_classes
    data_cfg = config["data"]
    images = generator.sample(
        num_synthetic_samples,
        num_inference_steps=gen_cfg.get("num_inference_steps"),
        batch_size=data_cfg["batch_size"],
        class_labels=labels,
    )
    synthetic_dataset = InMemoryDataset(images.detach().cpu(), labels, num_classes=num_classes)
    synthetic_loader = DataLoader(
        synthetic_dataset, batch_size=data_cfg["batch_size"], shuffle=True
    )

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
        synthetic_loader,
        real_test_loader,
        num_epochs=clf_cfg["num_epochs"],
        lr=clf_cfg["lr"],
        device=device,
    )
    print(f"Train losses (on synthetic data): {result.train_losses}")
    print(f"Validation accuracy (on real MNIST test set): {result.val_accuracy}")

    torch.save(model.state_dict(), output_dir / f"classifier_on_synthetic{suffix}.pt")
    save_config(config, output_dir / f"classifier_on_synthetic{suffix}_config.yaml")
    save_config(
        {
            "train_losses": result.train_losses,
            "val_accuracy": result.val_accuracy,
            "num_synthetic_train_samples": num_synthetic_samples,
        },
        output_dir / f"classifier_on_synthetic{suffix}_metrics.yaml",
    )
    print(f"Saved classifier, metrics, and config to {output_dir}")

    real_metrics_path = output_dir / "classifier_metrics.yaml"
    if real_metrics_path.exists():
        real_accuracy = load_config(real_metrics_path)["val_accuracy"]
        print(
            f"Comparison: real-data classifier {real_accuracy:.4f} vs. "
            f"synthetic-data classifier {result.val_accuracy:.4f} "
            "(both evaluated on the real MNIST test set)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--label",
        default=None,
        help="Optional tag for the output filenames (classifier_on_synthetic_<label>*).",
    )
    args = parser.parse_args()
    main(args.config, args.label)
