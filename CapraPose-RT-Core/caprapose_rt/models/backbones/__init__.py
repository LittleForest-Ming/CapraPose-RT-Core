"""Backbone exports for the public core package."""

from .rtmpose_m import RTMPoseMBackbone
from .rtmpose_style import RTMPoseStyleBackbone


def build_backbone(backbone_cfg: dict):
    """Build the lightweight visual encoder used by the core skeleton."""

    variant = str(backbone_cfg.get("variant", "RTMPose-m")).strip()
    backbone_kwargs = dict(backbone_cfg)
    backbone_kwargs.pop("variant", None)
    if variant in {"RTMPose-m", "CapraPose-LiteBackbone"}:
        return RTMPoseMBackbone(**backbone_kwargs)
    raise ValueError(
        "Unsupported public-core backbone variant: "
        f"{variant}. Expected 'RTMPose-m' or 'CapraPose-LiteBackbone'."
    )


__all__ = ["RTMPoseMBackbone", "RTMPoseStyleBackbone", "build_backbone"]
