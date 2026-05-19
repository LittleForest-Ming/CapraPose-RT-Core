"""Model builder for the public core package."""

from .pose_estimator import GoatPoseEstimator


def build_model(config: dict) -> GoatPoseEstimator:
    """Build CapraPose-RT from a config dictionary."""

    return GoatPoseEstimator(
        num_keypoints=int(config["dataset"]["num_keypoints"]),
        input_size=tuple(config["dataset"]["input_size"]),
        heatmap_size=tuple(config["dataset"]["heatmap_size"]),
        backbone_cfg=config["model"]["backbone"],
        decoder_cfg=config["model"]["decoder"],
        head_cfg=config["model"]["head"],
        refinement_cfg=config["model"]["refinement"],
        loss_cfg=config["loss"],
    )


__all__ = ["GoatPoseEstimator", "build_model"]
