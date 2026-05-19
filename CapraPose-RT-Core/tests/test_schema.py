from caprapose_rt.constants import (
    ANGLE_TRIPLETS,
    BILATERAL_SYMMETRY_EDGES,
    FLIP_PAIRS,
    FORE_HIND_CORRESPONDENCE_EDGES,
    KEYPOINT_NAMES,
    LATENT_PART_GROUPS,
    NUM_KEYPOINTS,
    OKS_SIGMAS,
    SKELETON_EDGES,
    SYMMETRIC_BONE_PAIRS,
)
from caprapose_rt.schema import GOAT_POSE_17_SCHEMA


def test_schema_is_consistent() -> None:
    assert NUM_KEYPOINTS == 17
    assert KEYPOINT_NAMES == [
        "head_0",
        "head_1",
        "body_0",
        "body_1",
        "body_2",
        "forelimb_left_0",
        "forelimb_left_1",
        "forelimb_left_2",
        "forelimb_right_0",
        "forelimb_right_1",
        "forelimb_right_2",
        "hindlimb_left_0",
        "hindlimb_left_1",
        "hindlimb_left_2",
        "hindlimb_right_0",
        "hindlimb_right_1",
        "hindlimb_right_2",
    ]
    assert len(OKS_SIGMAS) == NUM_KEYPOINTS
    assert GOAT_POSE_17_SCHEMA.confirmed_keypoint_names == tuple(KEYPOINT_NAMES)
    assert GOAT_POSE_17_SCHEMA.provisional_keypoint_names == ()
    assert GOAT_POSE_17_SCHEMA.paired_chain_groups["forelimbs"] == ((5, 6, 7), (8, 9, 10))
    assert GOAT_POSE_17_SCHEMA.paired_chain_groups["hindlimbs"] == ((11, 12, 13), (14, 15, 16))

    grouped_indices = sorted(
        index
        for group_indices in LATENT_PART_GROUPS.values()
        for index in group_indices
    )
    assert grouped_indices == list(range(NUM_KEYPOINTS))

    for start_index, end_index in (
        list(SKELETON_EDGES)
        + list(BILATERAL_SYMMETRY_EDGES)
        + list(FORE_HIND_CORRESPONDENCE_EDGES)
        + list(FLIP_PAIRS)
    ):
        assert 0 <= start_index < NUM_KEYPOINTS
        assert 0 <= end_index < NUM_KEYPOINTS

    for edge_a, edge_b in SYMMETRIC_BONE_PAIRS:
        for start_index, end_index in (edge_a, edge_b):
            assert 0 <= start_index < NUM_KEYPOINTS
            assert 0 <= end_index < NUM_KEYPOINTS

    for first_index, second_index, third_index in ANGLE_TRIPLETS:
        assert 0 <= first_index < NUM_KEYPOINTS
        assert 0 <= second_index < NUM_KEYPOINTS
        assert 0 <= third_index < NUM_KEYPOINTS
