"""Stage 1 entry point: boundary-focused sampling + synthetic-data/classifier evaluation.

Loads the generator and classifier trained by ``train_generator.py`` and
``train_classifier.py``, draws boundary-focused synthetic samples, and
reports synthetic-data quality metrics plus classifier metrics on real data.

Usage:
    python experiments/evaluate.py --config configs/mnist_baseline.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pd_synth.classifiers import SimpleCNN
from pd_synth.data import get_dataset
from pd_synth.evaluation import (
    confusion_matrix,
    frechet_distance_diagonal,
    mean_image_difference,
    precision_recall_f1,
)
from pd_synth.generation import DiffusionConfig, DiffusionGenerator
from pd_synth.sampling import BoundaryFocusedSampler
from pd_synth.utils import load_config, save_config, set_seed


def main(config_path: str) -> None:
    """Run boundary-focused sampling and evaluation per ``config_path``."""
    config = load_config(config_path)
    set_seed(config["seed"])

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

    clf_cfg = config["classifier"]
    classifier = SimpleCNN(
        in_channels=clf_cfg["in_channels"],
        image_size=clf_cfg["image_size"],
        num_classes=clf_cfg["num_classes"],
    )
    classifier.load_state_dict(torch.load(output_dir / "classifier.pt", map_location=device))
    classifier.to(device)
    classifier.eval()

    data_cfg = config["data"]

    def generate_fn(n: int) -> torch.Tensor:
        return generator.sample(
            n,
            num_inference_steps=gen_cfg.get("num_inference_steps"),
            batch_size=data_cfg["batch_size"],
        )

    @torch.no_grad()
    def classify_fn(images: torch.Tensor) -> torch.Tensor:
        logits = classifier(images.to(device))
        return torch.softmax(logits, dim=1).cpu()

    sampling_cfg = config["sampling"]
    sampler = BoundaryFocusedSampler(
        generate_fn, classify_fn, oversample_factor=sampling_cfg["oversample_factor"]
    )
    synthetic_images, boundary_scores = sampler.sample(sampling_cfg["num_synthetic_samples"])

    real_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=False,
        download=data_cfg.get("download", True),
    )
    real_loader = DataLoader(real_dataset, batch_size=data_cfg["batch_size"])
    real_images = torch.cat([images for images, _labels in real_loader], dim=0)

    real_flat = real_images.flatten(start_dim=1).numpy()
    synthetic_flat = synthetic_images.detach().cpu().flatten(start_dim=1).numpy()

    quality_metrics = {
        "mean_image_difference": mean_image_difference(real_flat, synthetic_flat),
        "frechet_distance_diagonal": frechet_distance_diagonal(real_flat, synthetic_flat),
        "mean_boundary_score": float(boundary_scores.mean().item()),
    }

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in real_loader:
            preds = classifier(images.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    preds_t, labels_t = torch.cat(all_preds), torch.cat(all_labels)
    prf1 = precision_recall_f1(preds_t, labels_t, num_classes=clf_cfg["num_classes"])
    cm = confusion_matrix(preds_t, labels_t, num_classes=clf_cfg["num_classes"])

    report = {
        "synthetic_data_quality": quality_metrics,
        "classifier_precision": prf1["precision"].tolist(),
        "classifier_recall": prf1["recall"].tolist(),
        "classifier_f1": prf1["f1"].tolist(),
        "confusion_matrix": cm.tolist(),
    }
    save_config(report, output_dir / "evaluation_report.yaml")
    save_config(config, output_dir / "evaluation_config.yaml")
    print(f"Synthetic-data quality: {quality_metrics}")
    print(f"Saved evaluation report and config to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    main(args.config)
