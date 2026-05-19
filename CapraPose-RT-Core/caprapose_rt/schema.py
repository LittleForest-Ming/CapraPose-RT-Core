"""Canonical 17-keypoint schema for CapraPose-RT.

This module is the single source of truth for the dairy-goat 2D pose schema
used across dataset normalization, model topology, visualization, and metrics.

The final project convention keeps the 17-point Dataset1 annotation order
unchanged while promoting the bilateral limb chains to confirmed anatomical
laterality after manual human audit:

- `head_*` for the cranial chain
- `body_*` for the three-point axial/body chain
- `forelimb_left_*` / `forelimb_right_*` for the two forelimb chains
- `hindlimb_left_*` / `hindlimb_right_*` for the two hindlimb chains

The `head_*` and `body_*` labels remain compact project identifiers aligned to
annotation order, while the limb-side semantics are now official.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointDefinition:
    """Metadata for one canonical keypoint entry."""

    name: str
    status: str
    note: str


@dataclass(frozen=True)
class PoseSchema:
    """Immutable project-wide schema definition."""

    schema_name: str
    schema_version: str
    keypoints: tuple[KeypointDefinition, ...]
    latent_part_groups: dict[str, tuple[int, ...]]
    paired_chain_groups: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    skeleton_edges: tuple[tuple[int, int], ...]
    bilateral_symmetry_edges: tuple[tuple[int, int], ...]
    fore_hind_correspondence_edges: tuple[tuple[int, int], ...]
    flip_pairs: tuple[tuple[int, int], ...]
    symmetric_bone_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    angle_triplets: tuple[tuple[int, int, int], ...]
    oks_sigmas: tuple[float, ...]
    semantic_note: str
    skeleton_index_base: int = 0

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    @property
    def keypoint_names(self) -> tuple[str, ...]:
        return tuple(keypoint.name for keypoint in self.keypoints)

    @property
    def keypoint_status(self) -> tuple[str, ...]:
        return tuple(keypoint.status for keypoint in self.keypoints)

    @property
    def provisional_keypoint_names(self) -> tuple[str, ...]:
        return tuple(
            keypoint.name
            for keypoint in self.keypoints
            if keypoint.status == "provisional"
        )

    @property
    def confirmed_keypoint_names(self) -> tuple[str, ...]:
        return tuple(
            keypoint.name
            for keypoint in self.keypoints
            if keypoint.status == "confirmed"
        )

    def coco_category(
        self,
        category_id: int = 1,
        category_name: str = "dairy_goat_pose",
        supercategory: str = "dairy_goat",
    ) -> dict[str, object]:
        """Export canonical category metadata for normalized annotations."""

        return {
            "id": category_id,
            "name": category_name,
            "supercategory": supercategory,
            "keypoints": list(self.keypoint_names),
            "skeleton": [list(edge) for edge in self.skeleton_edges],
            "caprapose_schema_name": self.schema_name,
            "caprapose_schema_version": self.schema_version,
            "caprapose_keypoint_status": list(self.keypoint_status),
            "caprapose_keypoint_note": self.semantic_note,
            "caprapose_skeleton_index_base": self.skeleton_index_base,
        }

    def schema_manifest(self) -> dict[str, object]:
        """Serialize the schema into a JSON-friendly manifest."""

        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "num_keypoints": self.num_keypoints,
            "semantic_note": self.semantic_note,
            "keypoints": [
                {
                    "index": index,
                    "name": keypoint.name,
                    "status": keypoint.status,
                    "note": keypoint.note,
                }
                for index, keypoint in enumerate(self.keypoints)
            ],
            "latent_part_groups": {
                name: list(indices) for name, indices in self.latent_part_groups.items()
            },
            "paired_chain_groups": {
                name: [list(group_a), list(group_b)]
                for name, (group_a, group_b) in self.paired_chain_groups.items()
            },
            "skeleton_edges": [list(edge) for edge in self.skeleton_edges],
            "bilateral_symmetry_edges": [
                list(edge) for edge in self.bilateral_symmetry_edges
            ],
            "fore_hind_correspondence_edges": [
                list(edge) for edge in self.fore_hind_correspondence_edges
            ],
            "flip_pairs": [list(edge) for edge in self.flip_pairs],
            "symmetric_bone_pairs": [
                [list(edge_a), list(edge_b)]
                for edge_a, edge_b in self.symmetric_bone_pairs
            ],
            "angle_triplets": [list(triplet) for triplet in self.angle_triplets],
            "oks_sigmas": list(self.oks_sigmas),
            "skeleton_index_base": self.skeleton_index_base,
        }


GOAT_POSE_17_SCHEMA = PoseSchema(
    schema_name="caprapose_goat17_dataset1",
    schema_version="1.1",
    keypoints=(
        KeypointDefinition(
            name="head_0",
            status="confirmed",
            note="Official project label for the first head-chain point in annotation order.",
        ),
        KeypointDefinition(
            name="head_1",
            status="confirmed",
            note="Official project label for the second head-chain point in annotation order.",
        ),
        KeypointDefinition(
            name="body_0",
            status="confirmed",
            note="Official project label for the first point on the axial body chain.",
        ),
        KeypointDefinition(
            name="body_1",
            status="confirmed",
            note="Official project label for the second point on the axial body chain.",
        ),
        KeypointDefinition(
            name="body_2",
            status="confirmed",
            note="Official project label for the third point on the axial body chain.",
        ),
        KeypointDefinition(
            name="forelimb_left_0",
            status="confirmed",
            note="Confirmed proximal point on the left forelimb chain.",
        ),
        KeypointDefinition(
            name="forelimb_left_1",
            status="confirmed",
            note="Confirmed middle point on the left forelimb chain.",
        ),
        KeypointDefinition(
            name="forelimb_left_2",
            status="confirmed",
            note="Confirmed distal point on the left forelimb chain.",
        ),
        KeypointDefinition(
            name="forelimb_right_0",
            status="confirmed",
            note="Confirmed proximal point on the right forelimb chain.",
        ),
        KeypointDefinition(
            name="forelimb_right_1",
            status="confirmed",
            note="Confirmed middle point on the right forelimb chain.",
        ),
        KeypointDefinition(
            name="forelimb_right_2",
            status="confirmed",
            note="Confirmed distal point on the right forelimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_left_0",
            status="confirmed",
            note="Confirmed proximal point on the left hindlimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_left_1",
            status="confirmed",
            note="Confirmed middle point on the left hindlimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_left_2",
            status="confirmed",
            note="Confirmed distal point on the left hindlimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_right_0",
            status="confirmed",
            note="Confirmed proximal point on the right hindlimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_right_1",
            status="confirmed",
            note="Confirmed middle point on the right hindlimb chain.",
        ),
        KeypointDefinition(
            name="hindlimb_right_2",
            status="confirmed",
            note="Confirmed distal point on the right hindlimb chain.",
        ),
    ),
    latent_part_groups={
        "head_neck": (0, 1, 2),
        "trunk": (3, 4),
        "forelimbs": (5, 6, 7, 8, 9, 10),
        "hindlimbs": (11, 12, 13, 14, 15, 16),
    },
    paired_chain_groups={
        "forelimbs": ((5, 6, 7), (8, 9, 10)),
        "hindlimbs": ((11, 12, 13), (14, 15, 16)),
    },
    skeleton_edges=(
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (2, 5),
        (5, 6),
        (6, 7),
        (2, 8),
        (8, 9),
        (9, 10),
        (3, 11),
        (11, 12),
        (12, 13),
        (3, 14),
        (14, 15),
        (15, 16),
    ),
    bilateral_symmetry_edges=(
        (5, 8),
        (6, 9),
        (7, 10),
        (11, 14),
        (12, 15),
        (13, 16),
    ),
    fore_hind_correspondence_edges=(
        (5, 11),
        (6, 12),
        (7, 13),
        (8, 14),
        (9, 15),
        (10, 16),
    ),
    flip_pairs=(
        (5, 8),
        (6, 9),
        (7, 10),
        (11, 14),
        (12, 15),
        (13, 16),
    ),
    symmetric_bone_pairs=(
        ((2, 5), (2, 8)),
        ((5, 6), (8, 9)),
        ((6, 7), (9, 10)),
        ((3, 11), (3, 14)),
        ((11, 12), (14, 15)),
        ((12, 13), (15, 16)),
    ),
    angle_triplets=(
        (1, 2, 3),
        (2, 5, 6),
        (5, 6, 7),
        (2, 8, 9),
        (8, 9, 10),
        (3, 11, 12),
        (11, 12, 13),
        (3, 14, 15),
        (14, 15, 16),
    ),
    oks_sigmas=(
        0.05,
        0.05,
        0.06,
        0.06,
        0.07,
        0.07,
        0.08,
        0.08,
        0.07,
        0.08,
        0.08,
        0.07,
        0.08,
        0.08,
        0.07,
        0.08,
        0.08,
    ),
    semantic_note=(
        "CapraPose-RT uses a finalized 17-keypoint schema aligned to Dataset1 annotation order. "
        "Manual human audit confirmed that the first forelimb and hindlimb chains are the left side, "
        "and the second forelimb and hindlimb chains are the right side. "
        "The head_* and body_* identifiers remain the official project labels for the non-lateral axial chain."
    ),
    skeleton_index_base=0,
)


def _validate_schema(schema: PoseSchema) -> None:
    if len(set(schema.keypoint_names)) != schema.num_keypoints:
        raise ValueError("Schema keypoint names must be unique.")
    if len(schema.oks_sigmas) != schema.num_keypoints:
        raise ValueError("Schema OKS sigmas must match the number of keypoints.")

    def _validate_index(index: int) -> None:
        if index < 0 or index >= schema.num_keypoints:
            raise ValueError(
                f"Schema index {index} is out of range [0, {schema.num_keypoints - 1}]."
            )

    for edge_group in (
        schema.skeleton_edges,
        schema.bilateral_symmetry_edges,
        schema.fore_hind_correspondence_edges,
        schema.flip_pairs,
    ):
        for start_index, end_index in edge_group:
            _validate_index(start_index)
            _validate_index(end_index)

    for edge_a, edge_b in schema.symmetric_bone_pairs:
        for start_index, end_index in (edge_a, edge_b):
            _validate_index(start_index)
            _validate_index(end_index)

    for first_index, second_index, third_index in schema.angle_triplets:
        _validate_index(first_index)
        _validate_index(second_index)
        _validate_index(third_index)

    grouped_indices = sorted(
        joint_index
        for group_indices in schema.latent_part_groups.values()
        for joint_index in group_indices
    )
    if grouped_indices != list(range(schema.num_keypoints)):
        raise ValueError(
            "latent_part_groups must cover every joint index exactly once. "
            f"Got {grouped_indices}."
        )


_validate_schema(GOAT_POSE_17_SCHEMA)
