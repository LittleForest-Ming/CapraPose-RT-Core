"""Topology-consistent structural learning losses."""

from __future__ import annotations

import torch
from torch import nn

from caprapose_rt.constants import ANGLE_TRIPLETS, ANATOMICAL_EDGES, SYMMETRIC_BONE_PAIRS
from caprapose_rt.utils.geometry import edge_length, joint_angle


class TopologyConsistentStructuralLoss(nn.Module):
    """Structural losses for the CapraPose-RT paper modules."""

    def __init__(
        self,
        bone_ratio_weight: float,
        symmetry_weight: float,
        angle_weight: float,
        torso_edge: tuple[int, int] = (2, 3),
    ) -> None:
        super().__init__()
        self.bone_ratio_weight = bone_ratio_weight
        self.symmetry_weight = symmetry_weight
        self.angle_weight = angle_weight
        self.torso_edge = torso_edge

    def forward(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bone_ratio = (
            self._bone_ratio_consistency(pred_coords, gt_coords, visibility)
            if self.bone_ratio_weight > 0
            else pred_coords.new_tensor(0.0)
        )
        symmetry = (
            self._bilateral_symmetry_consistency(pred_coords, gt_coords, visibility)
            if self.symmetry_weight > 0
            else pred_coords.new_tensor(0.0)
        )
        angle = (
            self._angle_smoothness(pred_coords, gt_coords, visibility)
            if self.angle_weight > 0
            else pred_coords.new_tensor(0.0)
        )
        total = (
            self.bone_ratio_weight * bone_ratio
            + self.symmetry_weight * symmetry
            + self.angle_weight * angle
        )
        return {
            "bone_ratio": bone_ratio,
            "symmetry": symmetry,
            "angle": angle,
            "total": total,
        }

    def _bone_ratio_consistency(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        pred_torso = edge_length(pred_coords, self.torso_edge).clamp_min(1e-6)
        gt_torso = edge_length(gt_coords, self.torso_edge).clamp_min(1e-6)

        loss_terms = []
        for edge in ANATOMICAL_EDGES:
            valid = (
                visibility[:, edge[0]]
                * visibility[:, edge[1]]
                * visibility[:, self.torso_edge[0]]
                * visibility[:, self.torso_edge[1]]
            )
            pred_ratio = edge_length(pred_coords, edge) / pred_torso
            gt_ratio = edge_length(gt_coords, edge) / gt_torso
            loss_terms.append(self._masked_mean((pred_ratio - gt_ratio).abs(), valid))
        return torch.stack(loss_terms).mean()

    def _bilateral_symmetry_consistency(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        pred_torso = edge_length(pred_coords, self.torso_edge).clamp_min(1e-6)
        gt_torso = edge_length(gt_coords, self.torso_edge).clamp_min(1e-6)

        loss_terms = []
        for edge_a, edge_b in SYMMETRIC_BONE_PAIRS:
            valid = (
                visibility[:, edge_a[0]]
                * visibility[:, edge_a[1]]
                * visibility[:, edge_b[0]]
                * visibility[:, edge_b[1]]
            )
            pred_delta = (
                edge_length(pred_coords, edge_a) - edge_length(pred_coords, edge_b)
            ).abs() / pred_torso
            gt_delta = (
                edge_length(gt_coords, edge_a) - edge_length(gt_coords, edge_b)
            ).abs() / gt_torso
            loss_terms.append(self._masked_mean((pred_delta - gt_delta).abs(), valid))
        return torch.stack(loss_terms).mean()

    def _angle_smoothness(
        self,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        loss_terms = []
        for triplet in ANGLE_TRIPLETS:
            valid = visibility[:, triplet[0]] * visibility[:, triplet[1]] * visibility[:, triplet[2]]
            pred_angle = joint_angle(pred_coords, triplet)
            gt_angle = joint_angle(gt_coords, triplet)
            loss_terms.append(self._masked_mean((pred_angle - gt_angle).abs(), valid))
        return torch.stack(loss_terms).mean()

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.float()
        return (values * weights).sum() / weights.sum().clamp_min(1.0)
