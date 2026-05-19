"""Compact RTMPose-m baseline backbone used by CapraPose-RT.

This repository keeps the first version self-contained, so this module is a
readable PyTorch approximation of the RTMPose-m design intent rather than a
verbatim copy of the official MMPose implementation. Swap this module with an
official RTMPose-m backbone when exact benchmark parity is required.
"""

from __future__ import annotations

from caprapose_rt.models.backbones.rtmpose_style import RTMPoseStyleBackbone


class RTMPoseMBackbone(RTMPoseStyleBackbone):
    """Alias the compact RTM-style backbone as the repo's RTMPose-m baseline."""

