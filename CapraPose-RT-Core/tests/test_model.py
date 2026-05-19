import torch

from caprapose_rt.constants import KEYPOINT_NAMES
from caprapose_rt.models import build_model


def _make_config(
    decoder_enabled: bool = True,
    refinement_enabled: bool = True,
    structural_weight: float = 0.2,
) -> dict:
    return {
        "dataset": {
            "num_keypoints": len(KEYPOINT_NAMES),
            "input_size": [256, 256],
            "heatmap_size": [64, 64],
        },
        "model": {
            "backbone": {
                "in_channels": 3,
                "stem_channels": 32,
                "stage_channels": [64, 96, 128],
                "num_blocks": [2, 2, 2],
            },
            "decoder": {
                "enabled": decoder_enabled,
                "hidden_channels": 160,
                "part_token_dim": 160,
                "joint_token_dim": 160,
                "dropout": 0.0,
            },
            "head": {"hidden_channels": 128},
            "refinement": {
                "enabled": refinement_enabled,
                "feature_dim": 128,
                "hidden_dim": 32,
                "step_size": 0.15,
            },
        },
        "loss": {
            "heatmap_weight": 1.0,
            "structural_weight": structural_weight,
            "bone_ratio_weight": 0.4,
            "symmetry_weight": 0.3,
            "angle_weight": 0.3,
        },
    }


def test_model_forward_full_caprapose_rt() -> None:
    config = _make_config(decoder_enabled=True, refinement_enabled=True)
    model = build_model(config)
    images = torch.randn(2, 3, 256, 256)
    outputs = model(images)

    assert outputs["heatmaps"].shape == (2, len(KEYPOINT_NAMES), 64, 64)
    assert outputs["refined_coords"].shape == (2, len(KEYPOINT_NAMES), 2)
    assert outputs["confidence"].shape == (2, len(KEYPOINT_NAMES), 1)
    assert outputs["joint_features"].shape == (2, len(KEYPOINT_NAMES), 128)
    assert outputs["joint_tokens"].shape == (2, len(KEYPOINT_NAMES), 160)
    assert set(outputs["part_tokens"].keys()) == {
        "head_neck",
        "trunk",
        "forelimbs",
        "hindlimbs",
    }
    assert outputs["refinement_diagnostics"]["anatomical"].shape[1] > 0


def test_model_forward_baseline_variant() -> None:
    config = _make_config(decoder_enabled=False, refinement_enabled=False, structural_weight=0.0)
    model = build_model(config)
    images = torch.randn(1, 3, 256, 256)
    outputs = model(images)

    assert outputs["heatmaps"].shape == (1, len(KEYPOINT_NAMES), 64, 64)
    assert outputs["joint_features"].shape == (1, len(KEYPOINT_NAMES), 128)
    assert outputs["joint_tokens"] is None
    assert outputs["part_tokens"] == {}


def test_model_loss_computation() -> None:
    config = _make_config(decoder_enabled=True, refinement_enabled=True, structural_weight=0.2)
    model = build_model(config)
    images = torch.randn(2, 3, 256, 256)
    outputs = model(images)
    batch = {
        "heatmaps": torch.randn(2, len(KEYPOINT_NAMES), 64, 64),
        "target_weight": torch.ones(2, len(KEYPOINT_NAMES), 1),
        "keypoints": torch.rand(2, len(KEYPOINT_NAMES), 2) * 256.0,
        "visibility": torch.ones(2, len(KEYPOINT_NAMES)),
    }
    losses = model.compute_losses(outputs, batch, config["loss"])
    assert torch.isfinite(losses["loss"])
    assert losses["structural_loss"].item() >= 0.0
