"""Shared constants for the dairy-goat 2D pose task.

This module re-exports the canonical schema from :mod:`caprapose_rt.schema`
for compatibility with the existing codebase.
"""

from __future__ import annotations

from caprapose_rt.schema import GOAT_POSE_17_SCHEMA

KEYPOINT_NAMES = list(GOAT_POSE_17_SCHEMA.keypoint_names)
KEYPOINT_SEMANTIC_NOTE = GOAT_POSE_17_SCHEMA.semantic_note

NUM_KEYPOINTS = len(KEYPOINT_NAMES)
KEYPOINT_NAME_TO_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

LATENT_PART_GROUPS = {
    part_name: list(indices)
    for part_name, indices in GOAT_POSE_17_SCHEMA.latent_part_groups.items()
}
PART_NAMES = list(LATENT_PART_GROUPS.keys())
JOINT_TO_PART = {
    joint_index: part_name
    for part_name, joint_indices in LATENT_PART_GROUPS.items()
    for joint_index in joint_indices
}

SKELETON_EDGES = list(GOAT_POSE_17_SCHEMA.skeleton_edges)
ANATOMICAL_EDGES = SKELETON_EDGES

BILATERAL_SYMMETRY_EDGES = list(GOAT_POSE_17_SCHEMA.bilateral_symmetry_edges)

FORE_HIND_CORRESPONDENCE_EDGES = list(
    GOAT_POSE_17_SCHEMA.fore_hind_correspondence_edges
)
RELATION_EDGE_GROUPS = {
    "anatomical": ANATOMICAL_EDGES,
    "symmetry": BILATERAL_SYMMETRY_EDGES,
    "correspondence": FORE_HIND_CORRESPONDENCE_EDGES,
}

FLIP_PAIRS = list(GOAT_POSE_17_SCHEMA.flip_pairs)

SYMMETRIC_BONE_PAIRS = list(GOAT_POSE_17_SCHEMA.symmetric_bone_pairs)

ANGLE_TRIPLETS = list(GOAT_POSE_17_SCHEMA.angle_triplets)

# The sigmas are heuristic placeholders for the initial release.
# Replace them with dataset-specific calibration for final benchmarking.
OKS_SIGMAS = list(GOAT_POSE_17_SCHEMA.oks_sigmas)

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def _validate_schema() -> None:
    if len(set(KEYPOINT_NAMES)) != NUM_KEYPOINTS:
        raise ValueError("KEYPOINT_NAMES must be unique.")
    if len(OKS_SIGMAS) != NUM_KEYPOINTS:
        raise ValueError(
            f"OKS_SIGMAS must have {NUM_KEYPOINTS} values, got {len(OKS_SIGMAS)}."
        )

    def _validate_index(index: int) -> None:
        if index < 0 or index >= NUM_KEYPOINTS:
            raise ValueError(f"Schema index {index} is out of range [0, {NUM_KEYPOINTS - 1}].")

    for edge_group in (
        SKELETON_EDGES,
        BILATERAL_SYMMETRY_EDGES,
        FORE_HIND_CORRESPONDENCE_EDGES,
        FLIP_PAIRS,
    ):
        for start_index, end_index in edge_group:
            _validate_index(start_index)
            _validate_index(end_index)

    for edge_a, edge_b in SYMMETRIC_BONE_PAIRS:
        for start_index, end_index in (edge_a, edge_b):
            _validate_index(start_index)
            _validate_index(end_index)

    for first_index, second_index, third_index in ANGLE_TRIPLETS:
        _validate_index(first_index)
        _validate_index(second_index)
        _validate_index(third_index)

    grouped_indices = sorted(
        joint_index
        for group_indices in LATENT_PART_GROUPS.values()
        for joint_index in group_indices
    )
    if grouped_indices != list(range(NUM_KEYPOINTS)):
        raise ValueError(
            "LATENT_PART_GROUPS must cover every joint index exactly once. "
            f"Got {grouped_indices}."
        )


_validate_schema()
