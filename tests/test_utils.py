"""Smoke tests for pd_synth.utils."""

from __future__ import annotations

from pathlib import Path

import torch

from pd_synth.utils.config import load_config, save_config
from pd_synth.utils.seed import set_seed


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = torch.rand(5)
    set_seed(123)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_config_round_trip(tmp_path: Path) -> None:
    config = {"seed": 1, "data": {"name": "mnist", "batch_size": 4}}
    path = tmp_path / "sub" / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
