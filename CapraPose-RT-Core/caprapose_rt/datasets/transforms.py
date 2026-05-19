"""Dataset preprocessing and target generation."""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from caprapose_rt.constants import FLIP_PAIRS, IMAGE_MEAN, IMAGE_STD, NUM_KEYPOINTS

IMAGE_MEAN_ARRAY = np.asarray(IMAGE_MEAN, dtype=np.float32)
IMAGE_STD_ARRAY = np.asarray(IMAGE_STD, dtype=np.float32)
PIL_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _expand_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    scale_factor: float,
) -> tuple[float, float, float, float]:
    x_coord, y_coord, width, height = bbox
    center_x = x_coord + width / 2.0
    center_y = y_coord + height / 2.0
    expanded_width = max(width * scale_factor, 2.0)
    expanded_height = max(height * scale_factor, 2.0)

    x1 = max(0.0, center_x - expanded_width / 2.0)
    y1 = max(0.0, center_y - expanded_height / 2.0)
    x2 = min(float(image_width), center_x + expanded_width / 2.0)
    y2 = min(float(image_height), center_y + expanded_height / 2.0)
    return x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _crop_and_resize(
    image: Image.Image,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    crop_box: Sequence[float],
    input_size: Sequence[int],
) -> tuple[Image.Image, np.ndarray]:
    crop_x, crop_y, crop_w, crop_h = crop_box
    left = int(math.floor(crop_x))
    top = int(math.floor(crop_y))
    right = int(math.ceil(crop_x + crop_w))
    bottom = int(math.ceil(crop_y + crop_h))

    crop = image.crop((left, top, right, bottom))
    resized = crop.resize(tuple(input_size), PIL_BILINEAR)

    transformed = keypoints.copy()
    transformed[:, 0] = (transformed[:, 0] - left) * (input_size[0] / max(right - left, 1))
    transformed[:, 1] = (transformed[:, 1] - top) * (input_size[1] / max(bottom - top, 1))
    transformed[visibility <= 0] = 0.0
    return resized, transformed


def _horizontal_flip(
    image: Image.Image,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    input_size: Sequence[int],
) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    flipped = ImageOps.mirror(image)
    flipped_keypoints = keypoints.copy()
    flipped_visibility = visibility.copy()
    flipped_keypoints[:, 0] = input_size[0] - 1 - flipped_keypoints[:, 0]

    for pair_a_idx, pair_b_idx in FLIP_PAIRS:
        flipped_keypoints[[pair_a_idx, pair_b_idx]] = flipped_keypoints[[pair_b_idx, pair_a_idx]]
        flipped_visibility[[pair_a_idx, pair_b_idx]] = flipped_visibility[[pair_b_idx, pair_a_idx]]

    return flipped, flipped_keypoints, flipped_visibility


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGE_MEAN_ARRAY) / IMAGE_STD_ARRAY
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class GoatPoseTransform:
    """Crop, normalize, and rasterize goat keypoints into heatmaps."""

    def __init__(
        self,
        input_size: Sequence[int],
        heatmap_size: Sequence[int],
        sigma: float,
        bbox_scale_factor: float,
        is_train: bool,
        flip_prob: float = 0.0,
    ) -> None:
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = sigma
        self.bbox_scale_factor = bbox_scale_factor
        self.is_train = is_train
        self.flip_prob = flip_prob

        self.input_width, self.input_height = self.input_size
        self.heatmap_width, self.heatmap_height = self.heatmap_size
        self.stride_x = self.input_width / self.heatmap_width
        self.stride_y = self.input_height / self.heatmap_height
        self.x_grid = np.arange(self.heatmap_width, dtype=np.float32)
        self.y_grid = np.arange(self.heatmap_height, dtype=np.float32)[:, None]

    def __call__(
        self,
        image: Image.Image,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        bbox: Sequence[float],
    ) -> dict[str, torch.Tensor]:
        image_width, image_height = image.size
        crop_box = _expand_bbox(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            scale_factor=self.bbox_scale_factor,
        )
        crop_image, crop_keypoints = _crop_and_resize(
            image=image,
            keypoints=keypoints,
            visibility=visibility,
            crop_box=crop_box,
            input_size=self.input_size,
        )

        if self.is_train and random.random() < self.flip_prob:
            crop_image, crop_keypoints, visibility = _horizontal_flip(
                image=crop_image,
                keypoints=crop_keypoints,
                visibility=visibility,
                input_size=self.input_size,
            )

        heatmaps, target_weight = self._generate_heatmaps(
            keypoints=crop_keypoints,
            visibility=visibility,
        )

        return {
            "image": _image_to_tensor(crop_image),
            "heatmaps": heatmaps,
            "target_weight": target_weight,
            "keypoints": torch.from_numpy(crop_keypoints.astype(np.float32)),
            "visibility": torch.from_numpy(visibility.astype(np.float32)),
            "crop_box": torch.tensor(crop_box, dtype=torch.float32),
        }

    def _generate_heatmaps(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heatmaps = np.zeros(
            (NUM_KEYPOINTS, self.heatmap_height, self.heatmap_width),
            dtype=np.float32,
        )
        target_weight = visibility.astype(np.float32).reshape(NUM_KEYPOINTS, 1)

        for index in range(NUM_KEYPOINTS):
            if visibility[index] <= 0:
                continue

            mu_x = keypoints[index, 0] / self.stride_x
            mu_y = keypoints[index, 1] / self.stride_y
            if (
                mu_x < 0
                or mu_y < 0
                or mu_x >= self.heatmap_width
                or mu_y >= self.heatmap_height
            ):
                target_weight[index] = 0.0
                continue

            exponent = (
                (self.x_grid - mu_x) ** 2 + (self.y_grid - mu_y) ** 2
            ) / (2.0 * self.sigma**2)
            heatmaps[index] = np.exp(-exponent)

        return torch.from_numpy(heatmaps), torch.from_numpy(target_weight)


def prepare_inference_sample(
    image: Image.Image,
    input_size: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Prepare a single full-image crop for inference."""

    width, height = image.size
    transform = GoatPoseTransform(
        input_size=input_size,
        heatmap_size=(input_size[0] // 4, input_size[1] // 4),
        sigma=2.5,
        bbox_scale_factor=1.0,
        is_train=False,
    )

    dummy_keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    dummy_visibility = np.zeros((NUM_KEYPOINTS,), dtype=np.float32)
    sample = transform(
        image=image,
        keypoints=dummy_keypoints,
        visibility=dummy_visibility,
        bbox=(0.0, 0.0, float(width), float(height)),
    )
    sample["original_size"] = torch.tensor([width, height], dtype=torch.float32)
    return sample
