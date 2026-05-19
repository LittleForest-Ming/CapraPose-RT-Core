"""Heatmap prediction head."""

from __future__ import annotations

import torch
from torch import nn

from caprapose_rt.models.backbones.rtmpose_style import ConvBNAct


class HeatmapHead(nn.Module):
    """Predict keypoint heatmaps from backbone or decoder features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_keypoints: int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, num_keypoints, kernel_size=1, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

