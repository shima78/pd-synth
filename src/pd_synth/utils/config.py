"""Config loading/saving so every run is config-driven and reproducible.

No hyperparameter or filesystem path should be hardcoded in pipeline code -
it should come from a YAML config in ``configs/`` and be persisted next to
that run's results via :func:`save_config`, so any result can later be
traced back to the exact config that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config from a YAML file.

    Args:
        path: Path to a YAML config file, typically under ``configs/``.

    Returns:
        The parsed config as a dictionary.

    Raises:
        ValueError: If the file does not parse to a mapping.
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping (got {type(config)!r})")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a config dict to YAML, creating parent directories as needed.

    Call this at the start of every experiment run so the exact config used
    is logged next to that run's outputs (e.g. ``outputs/<run>/config.yaml``).

    Args:
        config: The config dictionary to persist.
        path: Destination YAML path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
