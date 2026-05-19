"""Template config for the public CapraPose-RT core package.

The values wrapped in angle brackets are placeholders. They intentionally do
not include private dataset paths, trained checkpoints, experiment names, or
paper-specific hyperparameter sweeps.
"""

CONFIG = {
    "dataset": {
        "num_keypoints": 17,
        "input_size": [256, 256],
        "heatmap_size": [64, 64],
        "train_annotations": "<PATH_TO_TRAIN_COCO_JSON>",
        "val_annotations": "<PATH_TO_VAL_COCO_JSON>",
        "test_annotations": "<PATH_TO_TEST_COCO_JSON>",
        "image_root": "<PATH_TO_IMAGE_ROOT>",
    },
    "model": {
        "backbone": {
            "variant": "RTMPose-m",
            "in_channels": 3,
            "stem_channels": "<STEM_CHANNELS>",
            "stage_channels": ["<STAGE1_CHANNELS>", "<STAGE2_CHANNELS>", "<STAGE3_CHANNELS>"],
            "num_blocks": ["<STAGE1_BLOCKS>", "<STAGE2_BLOCKS>", "<STAGE3_BLOCKS>"],
        },
        "decoder": {
            "enabled": True,
            "implementation": "placeholder_in_public_release",
            "hidden_channels": "<DECODER_HIDDEN_CHANNELS>",
            "part_token_dim": "<PART_TOKEN_DIM>",
            "joint_token_dim": "<JOINT_TOKEN_DIM>",
            "dropout": "<DECODER_DROPOUT>",
        },
        "head": {
            "hidden_channels": "<HEATMAP_HEAD_HIDDEN_CHANNELS>",
        },
        "refinement": {
            "enabled": True,
            "feature_dim": "<JOINT_FEATURE_DIM>",
            "hidden_dim": "<TOPOLOGY_REFINEMENT_HIDDEN_DIM>",
            "step_size": "<REFINEMENT_STEP_SIZE>",
        },
    },
    "loss": {
        "heatmap_weight": 1.0,
        "refinement_coord_weight": "<REFINEMENT_COORD_WEIGHT>",
        "structural_weight": "<STRUCTURAL_LOSS_WEIGHT>",
        "bone_ratio_weight": "<BONE_RATIO_WEIGHT>",
        "symmetry_weight": "<SYMMETRY_WEIGHT>",
        "angle_weight": "<ANGLE_WEIGHT>",
    },
}
