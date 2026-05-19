"""Common engine helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def move_batch_to_device(
    batch: Mapping,
    device: torch.device,
    non_blocking: bool = True,
    channels_last: bool = False,
) -> dict:
    """Move tensor fields onto the target device while preserving metadata."""

    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if (
                channels_last
                and key == "image"
                and value.ndim == 4
                and device.type == "cuda"
            ):
                output[key] = value.to(
                    device,
                    non_blocking=non_blocking,
                    memory_format=torch.channels_last,
                )
            else:
                output[key] = value.to(device, non_blocking=non_blocking)
        else:
            output[key] = value
    return output
