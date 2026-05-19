"""Run a minimal random-input forward pass for the public core package."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from caprapose_rt.models import build_model


CONFIG = {
    "dataset": {
        "num_keypoints": 17,
        "input_size": [256, 256],
        "heatmap_size": [64, 64],
    },
    "model": {
        "backbone": {
            "variant": "RTMPose-m",
            "in_channels": 3,
            "stem_channels": 32,
            "stage_channels": [64, 96, 128],
            "num_blocks": [2, 2, 2],
        },
        "decoder": {
            "enabled": True,
            "hidden_channels": 160,
            "part_token_dim": 160,
            "joint_token_dim": 160,
            "dropout": 0.0,
        },
        "head": {"hidden_channels": 128},
        "refinement": {
            "enabled": True,
            "feature_dim": 128,
            "hidden_dim": 32,
            "step_size": 0.05,
        },
    },
    "loss": {
        "heatmap_weight": 1.0,
        "refinement_coord_weight": 1.0,
        "structural_weight": 0.0,
        "bone_ratio_weight": 0.4,
        "symmetry_weight": 0.3,
        "angle_weight": 0.3,
    },
}


def main() -> None:
    model = build_model(CONFIG).eval()
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(images, return_intermediates=True)
    print("heatmaps:", tuple(outputs["heatmaps"].shape))
    print("refined_coords:", tuple(outputs["refined_coords"].shape))
    print("joint_features:", tuple(outputs["joint_features"].shape))
    print("decoder placeholder:", "placeholder_notice" in outputs["decoder_diagnostics"])


if __name__ == "__main__":
    main()
