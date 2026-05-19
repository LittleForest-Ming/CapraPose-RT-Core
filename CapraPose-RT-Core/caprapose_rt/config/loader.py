"""Lightweight Python config loader."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_module_from_path(config_path: str | Path) -> ModuleType:
    path = Path(config_path).resolve()
    repo_root = str(path.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    spec = importlib.util.spec_from_file_location("caprapose_rt_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a config file that exposes a top-level CONFIG dictionary."""

    module = _load_module_from_path(config_path)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config file does not define CONFIG: {config_path}")
    return deepcopy(module.CONFIG)


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Persist the resolved config to JSON for experiment reproducibility."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
