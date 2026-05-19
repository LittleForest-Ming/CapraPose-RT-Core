"""Public placeholder for the latent part-aware hierarchical decoder.

The full research decoder is intentionally not included in this GitHub core
package. This module preserves the public class name, constructor arguments,
forward signature, and tensor contract so readers can understand where the
decoder sits in the CapraPose-RT pipeline without exposing experiment-specific
implementation details.
"""

from __future__ import annotations

import torch
from torch import nn

from caprapose_rt.constants import PART_NAMES
from caprapose_rt.models.backbones.rtmpose_style import ConvBNAct


class LatentPartAwareHierarchicalDecoder(nn.Module):
    """Interface-compatible placeholder for the private decoder implementation.

    Replace this class with the full decoder implementation when reproducing
    the complete method. The placeholder performs a simple feature projection
    and returns zero-valued part/joint tokens with the expected shapes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_keypoints: int,
        part_token_dim: int = 0,
        joint_token_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.part_token_dim = int(part_token_dim)
        self.joint_token_dim = int(joint_token_dim)
        self.part_names = tuple(PART_NAMES)
        self.feature_projection = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=1),
            nn.Dropout2d(p=float(dropout)) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        features = self.feature_projection(inputs)
        batch_size = int(inputs.shape[0])
        device = inputs.device
        dtype = inputs.dtype

        part_features: dict[str, torch.Tensor] = {}
        part_tokens = {
            name: torch.zeros(batch_size, self.part_token_dim, device=device, dtype=dtype)
            for name in self.part_names
        }
        joint_tokens = torch.zeros(
            batch_size,
            self.num_keypoints,
            self.joint_token_dim,
            device=device,
            dtype=dtype,
        )
        diagnostics = {
            "placeholder_notice": torch.ones(batch_size, 1, device=device, dtype=dtype)
        }
        return features, part_features, part_tokens, joint_tokens, diagnostics
