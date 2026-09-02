"""Loading and caching of a trained pd-synth run for the demo app.

Everything here is UI-framework-agnostic except for the two Streamlit cache
decorators: the app only ever asks for a :class:`RunArtifacts` bundle plus a
few metrics dicts, and never touches the ``experiments/`` output layout
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from pd_synth.classifiers import SimpleCNN
from pd_synth.data import get_dataset, index_by_class
from pd_synth.generation import DiffusionConfig, DiffusionGenerator
from pd_synth.utils import load_config


@dataclass
class RunArtifacts:
    """Everything the app needs from one trained run."""

    config: dict[str, Any]
    output_dir: Path
    generator: DiffusionGenerator
    classifier: SimpleCNN
    real_dataset: Any
    indices_by_class: dict[int, list[int]]

    @property
    def generator_config(self) -> dict[str, Any]:
        return self.config["generator"]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_run(config_path: str) -> RunArtifacts:
    """Load the generator, classifier, and real test dataset for ``config_path``."""
    config = load_config(config_path)
    output_dir = Path(config["output"]["dir"])
    device = _device()

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
    classifier.to(device).eval()

    data_cfg = config["data"]
    real_dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=False,
        download=data_cfg.get("download", True),
    )
    return RunArtifacts(
        config=config,
        output_dir=output_dir,
        generator=generator,
        classifier=classifier,
        real_dataset=real_dataset,
        indices_by_class=index_by_class(real_dataset),
    )


@st.cache_data
def load_metrics(path: str) -> dict[str, Any] | None:
    """Return a metrics/report YAML as a dict, or ``None`` when it doesn't exist yet."""
    path = Path(path)
    return load_config(path) if path.exists() else None


def accuracy_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Collect classifier accuracy across every training set that has been run.

    Each row is ``{"training_set", "accuracy", "data"}`` and feeds
    :func:`app.charts.accuracy_chart`. Runs that haven't happened yet are
    simply skipped.
    """
    full_real = load_metrics(str(output_dir / "classifier_metrics.yaml"))
    rows: list[dict[str, Any]] = []
    if full_real is not None:
        rows.append(
            {"training_set": "60,000 real", "accuracy": full_real["val_accuracy"], "data": "real"}
        )

    sources = (
        ("classifier_on_real_subset_metrics.yaml", "num_real_train_samples", "real"),
        ("classifier_on_synthetic_metrics.yaml", "num_synthetic_train_samples", "synthetic"),
        ("classifier_on_synthetic_full_metrics.yaml", "num_synthetic_train_samples", "synthetic"),
    )
    for filename, count_key, data_kind in sources:
        metrics = load_metrics(str(output_dir / filename))
        if metrics is None:
            continue
        count = metrics[count_key]
        rows.append(
            {
                "training_set": f"{count:,} {data_kind}",
                "accuracy": metrics["val_accuracy"],
                "data": data_kind,
            }
        )
    return rows
