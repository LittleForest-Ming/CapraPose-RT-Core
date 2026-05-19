"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def _extract_state_dict(checkpoint: dict) -> dict:
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = False,
    logger=None,
) -> dict | None:
    """Load a checkpoint into the model if the file exists."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        if logger:
            logger.warning("Checkpoint not found: %s", checkpoint_path)
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if logger:
        logger.info("Loaded checkpoint from %s", checkpoint_path)
        if getattr(incompatible, "missing_keys", None):
            logger.info("Missing keys: %s", incompatible.missing_keys)
        if getattr(incompatible, "unexpected_keys", None):
            logger.info("Unexpected keys: %s", incompatible.unexpected_keys)
    return checkpoint


def save_checkpoint(state: dict, work_dir: str | Path, filename: str) -> Path:
    """Persist a training checkpoint to disk."""

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / filename
    torch.save(state, checkpoint_path)
    return checkpoint_path

