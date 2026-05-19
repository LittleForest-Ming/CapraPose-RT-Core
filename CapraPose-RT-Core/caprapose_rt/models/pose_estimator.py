"""Core CapraPose-RT pose estimator skeleton.

This public package keeps the model pathway understandable while excluding
private experiment details. The latent decoder is represented by an
interface-compatible placeholder in ``heads/latent_part_decoder.py``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from caprapose_rt.models.backbones import build_backbone
from caprapose_rt.models.heads import HeatmapHead, LatentPartAwareHierarchicalDecoder
from caprapose_rt.models.losses import TopologyConsistentStructuralLoss
from caprapose_rt.models.modules import AdaptiveTopologyRefinement
from caprapose_rt.utils.geometry import scale_coords, spatial_argmax_2d, spatial_soft_argmax_2d


class GoatPoseEstimator(nn.Module):
    """2D dairy-goat pose estimator with structural refinement hooks."""

    def __init__(
        self,
        num_keypoints: int,
        input_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        backbone_cfg: dict,
        decoder_cfg: dict,
        head_cfg: dict,
        refinement_cfg: dict,
        loss_cfg: dict,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.input_size = tuple(input_size)
        self.heatmap_size = tuple(heatmap_size)

        self.backbone = build_backbone(backbone_cfg)
        feature_channels = int(self.backbone.out_channels)
        decoder_joint_token_dim = 0

        self.decoder = None
        if decoder_cfg.get("enabled", False):
            decoder_hidden_channels = int(decoder_cfg["hidden_channels"])
            decoder_joint_token_dim = int(
                decoder_cfg.get("joint_token_dim", decoder_hidden_channels)
            )
            self.decoder = LatentPartAwareHierarchicalDecoder(
                in_channels=feature_channels,
                hidden_channels=decoder_hidden_channels,
                num_keypoints=self.num_keypoints,
                part_token_dim=int(decoder_cfg.get("part_token_dim", decoder_hidden_channels)),
                joint_token_dim=decoder_joint_token_dim,
                dropout=float(decoder_cfg.get("dropout", 0.0)),
            )
            feature_channels = decoder_hidden_channels

        self.heatmap_head = HeatmapHead(
            in_channels=feature_channels,
            hidden_channels=int(head_cfg["hidden_channels"]),
            num_keypoints=self.num_keypoints,
        )
        self.joint_feature_dim = int(
            refinement_cfg.get("feature_dim", head_cfg["hidden_channels"])
        )
        self.joint_feature_adapter = nn.Sequential(
            nn.Linear(feature_channels + decoder_joint_token_dim, self.joint_feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.joint_feature_dim, self.joint_feature_dim),
        )

        self.refinement = None
        if refinement_cfg.get("enabled", False):
            self.refinement = AdaptiveTopologyRefinement(
                feature_dim=self.joint_feature_dim,
                hidden_dim=int(refinement_cfg["hidden_dim"]),
                step_size=float(refinement_cfg["step_size"]),
            )

        self.structural_loss = TopologyConsistentStructuralLoss(
            bone_ratio_weight=float(loss_cfg.get("bone_ratio_weight", 0.0)),
            symmetry_weight=float(loss_cfg.get("symmetry_weight", 0.0)),
            angle_weight=float(loss_cfg.get("angle_weight", 0.0)),
        )

    def forward(
        self,
        images: torch.Tensor,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        backbone_features = self.backbone(images)
        features = backbone_features
        part_features: dict[str, torch.Tensor] = {}
        part_tokens: dict[str, torch.Tensor] = {}
        decoder_diagnostics: dict[str, torch.Tensor] = {}
        joint_tokens = None

        if self.decoder is not None:
            features, part_features, part_tokens, joint_tokens, decoder_diagnostics = self.decoder(
                features
            )
        decoded_features = features

        heatmaps = self.heatmap_head(features)
        if self.training:
            coords_heatmap, confidence = spatial_soft_argmax_2d(heatmaps)
        else:
            coords_heatmap, confidence = spatial_argmax_2d(heatmaps)
        coarse_coords = scale_coords(
            coords=coords_heatmap,
            from_size=self.heatmap_size,
            to_size=self.input_size,
        )

        sampled_joint_features = self._sample_joint_features(features, coords_heatmap)
        if joint_tokens is not None:
            joint_feature_inputs = torch.cat([sampled_joint_features, joint_tokens], dim=-1)
        else:
            joint_feature_inputs = sampled_joint_features
        joint_features = self.joint_feature_adapter(joint_feature_inputs)

        refined_coords = coarse_coords
        refined_joint_features = joint_features
        refinement_diagnostics: dict[str, torch.Tensor] = {}
        relation_encoder_diagnostics: dict[str, torch.Tensor] = {}
        if self.refinement is not None:
            (
                refined_coords,
                refined_joint_features,
                refinement_diagnostics,
                relation_encoder_diagnostics,
            ) = self.refinement(coarse_coords, confidence, joint_features)

        outputs = {
            "heatmaps": heatmaps,
            "coarse_coords": coarse_coords,
            "refined_coords": refined_coords,
            "confidence": confidence,
            "joint_features": refined_joint_features,
            "adapted_joint_features": joint_features,
            "joint_tokens": joint_tokens,
            "part_features": part_features,
            "part_tokens": part_tokens,
            "refinement_diagnostics": refinement_diagnostics,
            "relation_encoder_diagnostics": relation_encoder_diagnostics,
        }
        if return_intermediates:
            outputs["backbone_features"] = backbone_features
            outputs["decoded_features"] = decoded_features
            outputs["sampled_joint_features"] = sampled_joint_features
            outputs["coords_heatmap"] = coords_heatmap
            outputs["decoder_diagnostics"] = decoder_diagnostics
        return outputs

    def compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        loss_cfg: dict,
    ) -> dict[str, torch.Tensor]:
        heatmap_error = (outputs["heatmaps"] - batch["heatmaps"]) ** 2
        target_weight = batch["target_weight"]
        weighted_heatmap_error = heatmap_error * target_weight.unsqueeze(-1)
        normalizer = (
            target_weight.sum().clamp_min(1.0)
            * outputs["heatmaps"].shape[-1]
            * outputs["heatmaps"].shape[-2]
        )
        heatmap_loss = weighted_heatmap_error.sum() / normalizer

        refinement_weight = float(loss_cfg.get("refinement_coord_weight", 0.0))
        if self.refinement is not None and refinement_weight > 0:
            coord_normalizer = outputs["refined_coords"].new_tensor(
                [float(self.input_size[0]), float(self.input_size[1])]
            ).view(1, 1, 2)
            coord_error = F.smooth_l1_loss(
                outputs["refined_coords"] / coord_normalizer,
                batch["keypoints"] / coord_normalizer,
                reduction="none",
            )
            visibility_mask = batch["visibility"].unsqueeze(-1)
            coord_loss = (coord_error * visibility_mask).sum() / visibility_mask.sum().clamp_min(1.0)
        else:
            coord_loss = heatmap_loss.new_tensor(0.0)

        structural_weight = float(loss_cfg.get("structural_weight", 0.0))
        if structural_weight > 0:
            structure_terms = self.structural_loss(
                pred_coords=outputs["refined_coords"],
                gt_coords=batch["keypoints"],
                visibility=batch["visibility"],
            )
        else:
            zero = heatmap_loss.new_tensor(0.0)
            structure_terms = {"bone_ratio": zero, "symmetry": zero, "angle": zero, "total": zero}

        total_loss = (
            float(loss_cfg.get("heatmap_weight", 1.0)) * heatmap_loss
            + refinement_weight * coord_loss
            + structural_weight * structure_terms["total"]
        )
        return {
            "loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "coord_loss": coord_loss,
            "bone_ratio_loss": structure_terms["bone_ratio"],
            "symmetry_loss": structure_terms["symmetry"],
            "angle_loss": structure_terms["angle"],
            "structural_loss": structure_terms["total"],
        }

    @staticmethod
    def _sample_joint_features(
        feature_map: torch.Tensor,
        coords_heatmap: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = feature_map.shape
        normalized_coords = coords_heatmap.clone()
        normalized_coords[..., 0] = (
            normalized_coords[..., 0] / max(width - 1, 1) * 2.0 - 1.0
        )
        normalized_coords[..., 1] = (
            normalized_coords[..., 1] / max(height - 1, 1) * 2.0 - 1.0
        )
        sampling_grid = normalized_coords.view(batch_size, coords_heatmap.shape[1], 1, 2)
        sampled = F.grid_sample(
            feature_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)
