"""Geometry helpers for keypoint decoding and evaluation."""

from __future__ import annotations

from typing import Sequence

import torch


def spatial_soft_argmax_2d(
    heatmaps: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode regression heatmaps into differentiable coordinate predictions.

    CapraPose-RT trains its keypoint head with Gaussian heatmap regression, so
    the heatmap tensor represents non-negative response magnitudes rather than
    classification logits. Normalizing positive responses is therefore the
    correct expectation-based decode. A softmax fallback is only used when a
    joint heatmap is entirely non-positive.
    """

    batch_size, num_keypoints, height, width = heatmaps.shape
    flattened = heatmaps.view(batch_size, num_keypoints, -1) / max(float(temperature), 1e-6)
    positive_responses = flattened.clamp_min(0.0)
    response_sum = positive_responses.sum(dim=-1, keepdim=True)
    fallback_probabilities = torch.softmax(flattened, dim=-1)
    probabilities = torch.where(
        response_sum > 0,
        positive_responses / response_sum.clamp_min(1e-6),
        fallback_probabilities,
    )

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)

    x = (probabilities * x_grid).sum(dim=-1)
    y = (probabilities * y_grid).sum(dim=-1)
    coords = torch.stack([x, y], dim=-1)
    confidence = positive_responses.max(dim=-1).values.unsqueeze(-1)
    return coords, confidence


def spatial_argmax_2d(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode heatmaps with a standard argmax-style peak picker."""

    batch_size, num_keypoints, _, width = heatmaps.shape
    flattened = heatmaps.view(batch_size, num_keypoints, -1)
    peak_indices = flattened.argmax(dim=-1)
    x_coord = (peak_indices % width).to(dtype=heatmaps.dtype)
    y_coord = torch.div(peak_indices, width, rounding_mode="floor").to(dtype=heatmaps.dtype)
    coords = torch.stack([x_coord, y_coord], dim=-1)
    confidence = flattened.max(dim=-1).values.unsqueeze(-1)
    return coords, confidence


def scale_coords(
    coords: torch.Tensor,
    from_size: Sequence[int],
    to_size: Sequence[int],
) -> torch.Tensor:
    """Scale coordinates between heatmap space and image space."""

    from_width, from_height = float(from_size[0]), float(from_size[1])
    to_width, to_height = float(to_size[0]), float(to_size[1])
    scaled = coords.clone()
    scaled[..., 0] = scaled[..., 0] * (to_width / max(from_width, 1.0))
    scaled[..., 1] = scaled[..., 1] * (to_height / max(from_height, 1.0))
    return scaled


def crop_coords_to_image(
    coords: torch.Tensor,
    crop_box: torch.Tensor,
    input_size: Sequence[int],
) -> torch.Tensor:
    """Project coordinates from resized crop space back to original image space."""

    input_width, input_height = float(input_size[0]), float(input_size[1])
    crop_x = crop_box[..., 0].unsqueeze(-1)
    crop_y = crop_box[..., 1].unsqueeze(-1)
    crop_w = crop_box[..., 2].unsqueeze(-1)
    crop_h = crop_box[..., 3].unsqueeze(-1)

    output = coords.clone()
    output[..., 0] = output[..., 0] * (crop_w / max(input_width, 1.0)) + crop_x
    output[..., 1] = output[..., 1] * (crop_h / max(input_height, 1.0)) + crop_y
    return output


def edge_length(coords: torch.Tensor, edge: tuple[int, int]) -> torch.Tensor:
    """Compute Euclidean edge lengths for a batch of coordinate sets."""

    start, end = edge
    return torch.linalg.norm(coords[:, start] - coords[:, end], dim=-1)


def joint_angle(coords: torch.Tensor, triplet: tuple[int, int, int]) -> torch.Tensor:
    """Compute the angle at the middle keypoint of a triplet."""

    a, b, c = triplet
    vec_ab = coords[:, a] - coords[:, b]
    vec_cb = coords[:, c] - coords[:, b]
    numerator = (vec_ab * vec_cb).sum(dim=-1)
    denominator = (
        torch.linalg.norm(vec_ab, dim=-1) * torch.linalg.norm(vec_cb, dim=-1)
    ).clamp_min(1e-6)
    cosine = (numerator / denominator).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)
