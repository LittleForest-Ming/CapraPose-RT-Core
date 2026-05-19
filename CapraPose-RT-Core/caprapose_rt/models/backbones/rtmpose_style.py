"""Compact RTM-style building blocks for the repo's RTMPose-m baseline."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    """Small convenience block used throughout the repository."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """Mobile-style residual block used for the real-time baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)
        if self.use_residual:
            outputs = outputs + inputs
        return outputs


class RTMPoseStyleBackbone(nn.Module):
    """Readable RTM-style backbone with heatmap-friendly output resolution."""

    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        stage_channels: list[int],
        num_blocks: list[int],
    ) -> None:
        super().__init__()
        if len(stage_channels) != 3 or len(num_blocks) != 3:
            raise ValueError("Expected three stages for stage_channels and num_blocks.")

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2),
            ConvBNAct(stem_channels, stem_channels, kernel_size=3, stride=1),
        )

        self.stage1 = self._make_stage(
            in_channels=stem_channels,
            out_channels=stage_channels[0],
            num_blocks=num_blocks[0],
            downsample=True,
        )
        self.stage2 = self._make_stage(
            in_channels=stage_channels[0],
            out_channels=stage_channels[1],
            num_blocks=num_blocks[1],
            downsample=False,
        )
        self.stage3 = self._make_stage(
            in_channels=stage_channels[1],
            out_channels=stage_channels[2],
            num_blocks=num_blocks[2],
            downsample=False,
        )
        self.out_channels = stage_channels[-1]

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
    ) -> nn.Sequential:
        layers = [
            DepthwiseSeparableBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if downsample else 1,
            )
        ]
        for _ in range(num_blocks - 1):
            layers.append(
                DepthwiseSeparableBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        return outputs
