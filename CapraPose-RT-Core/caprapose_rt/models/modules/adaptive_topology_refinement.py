"""Adaptive topology refinement for CapraPose-RT."""

from __future__ import annotations

import torch
from torch import nn

from caprapose_rt.constants import RELATION_EDGE_GROUPS


class AdaptiveTopologyRefinement(nn.Module):
    """Refine joint predictions using adaptive structural relations.

    The module operates on joint-level representations and predicted coordinates.
    Each relation category has its own lightweight parameterization, while the
    edge strengths are predicted adaptively from the current joint evidence.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        step_size: float = 0.15,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.step_size = step_size
        self.edge_sets = dict(RELATION_EDGE_GROUPS)

        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        relation_input_dim = hidden_dim * 3 + 4
        self.relation_encoders = nn.ModuleDict(
            {
                relation_name: nn.Sequential(
                    nn.Linear(relation_input_dim, hidden_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for relation_name in self.edge_sets
            }
        )
        self.edge_strength_heads = nn.ModuleDict(
            {
                relation_name: nn.Linear(hidden_dim, 1)
                for relation_name in self.edge_sets
            }
        )
        self.coord_message_heads = nn.ModuleDict(
            {
                relation_name: nn.Linear(hidden_dim, 2)
                for relation_name in self.edge_sets
            }
        )
        self.feature_message_heads = nn.ModuleDict(
            {
                relation_name: nn.Linear(hidden_dim, hidden_dim)
                for relation_name in self.edge_sets
            }
        )
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.feature_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        coords: torch.Tensor,
        confidence: torch.Tensor,
        joint_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        projected_features = self.feature_projection(joint_features)
        coord_messages = torch.zeros_like(coords)
        feature_messages = torch.zeros_like(projected_features)
        counts = torch.zeros(
            coords.shape[0],
            coords.shape[1],
            1,
            device=coords.device,
            dtype=coords.dtype,
        )
        diagnostics: dict[str, list[torch.Tensor]] = {
            relation_name: [] for relation_name in self.edge_sets
        }
        encoder_diagnostics: dict[str, list[torch.Tensor]] = {
            relation_name: [] for relation_name in self.edge_sets
        }

        for relation_name, edges in self.edge_sets.items():
            relation_encoder = self.relation_encoders[relation_name]
            strength_head = self.edge_strength_heads[relation_name]
            coord_head = self.coord_message_heads[relation_name]
            feature_head = self.feature_message_heads[relation_name]

            for source_idx, target_idx in edges:
                self._update_edge(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    relation_encoder=relation_encoder,
                    strength_head=strength_head,
                    coord_head=coord_head,
                    feature_head=feature_head,
                    coords=coords,
                    confidence=confidence,
                    projected_features=projected_features,
                    coord_messages=coord_messages,
                    feature_messages=feature_messages,
                    counts=counts,
                    diagnostics=diagnostics[relation_name],
                    encoder_diagnostics=encoder_diagnostics[relation_name],
                )
                self._update_edge(
                    source_idx=target_idx,
                    target_idx=source_idx,
                    relation_encoder=relation_encoder,
                    strength_head=strength_head,
                    coord_head=coord_head,
                    feature_head=feature_head,
                    coords=coords,
                    confidence=confidence,
                    projected_features=projected_features,
                    coord_messages=coord_messages,
                    feature_messages=feature_messages,
                    counts=counts,
                    diagnostics=diagnostics[relation_name],
                    encoder_diagnostics=encoder_diagnostics[relation_name],
                )

        normalized_counts = counts.clamp_min(1e-6)
        aggregated_coord_messages = coord_messages / normalized_counts
        aggregated_feature_messages = feature_messages / normalized_counts
        refined_context = projected_features + aggregated_feature_messages

        node_gate = torch.sigmoid(
            self.node_gate(torch.cat([refined_context, confidence], dim=-1))
        )
        refined_coords = coords + self.step_size * node_gate * aggregated_coord_messages
        refined_joint_features = joint_features + self.feature_update(
            torch.cat([projected_features, refined_context], dim=-1)
        )
        diagnostics_output = {
            relation_name: torch.stack(relation_values, dim=1)
            if relation_values
            else torch.empty(
                coords.shape[0],
                0,
                1,
                device=coords.device,
                dtype=coords.dtype,
            )
            for relation_name, relation_values in diagnostics.items()
        }
        encoder_output = {
            relation_name: torch.stack(relation_values, dim=1)
            if relation_values
            else torch.empty(
                coords.shape[0],
                0,
                self.hidden_dim,
                device=coords.device,
                dtype=coords.dtype,
            )
            for relation_name, relation_values in encoder_diagnostics.items()
        }
        return refined_coords, refined_joint_features, diagnostics_output, encoder_output

    def _update_edge(
        self,
        source_idx: int,
        target_idx: int,
        relation_encoder: nn.Module,
        strength_head: nn.Module,
        coord_head: nn.Module,
        feature_head: nn.Module,
        coords: torch.Tensor,
        confidence: torch.Tensor,
        projected_features: torch.Tensor,
        coord_messages: torch.Tensor,
        feature_messages: torch.Tensor,
        counts: torch.Tensor,
        diagnostics: list[torch.Tensor],
        encoder_diagnostics: list[torch.Tensor],
    ) -> None:
        relation_features = self._relation_features(
            src_features=projected_features[:, source_idx],
            dst_features=projected_features[:, target_idx],
            src_coords=coords[:, source_idx],
            dst_coords=coords[:, target_idx],
            src_conf=confidence[:, source_idx],
            dst_conf=confidence[:, target_idx],
        )
        encoded_relation = relation_encoder(relation_features)
        strength = torch.sigmoid(strength_head(encoded_relation))
        # Keep refinement updates residual-sized. The refinement branch should
        # make gentle structural corrections on top of a working decoder rather
        # than learn unbounded coordinate rewrites that can destabilize stage-2
        # training.
        coord_update = torch.tanh(coord_head(encoded_relation))
        coord_messages[:, target_idx] = coord_messages[:, target_idx] + strength * coord_update
        feature_messages[:, target_idx] = feature_messages[:, target_idx] + strength * feature_head(
            encoded_relation
        )
        counts[:, target_idx] = counts[:, target_idx] + strength
        diagnostics.append(strength)
        encoder_diagnostics.append(encoded_relation)

    @staticmethod
    def _relation_features(
        src_features: torch.Tensor,
        dst_features: torch.Tensor,
        src_coords: torch.Tensor,
        dst_coords: torch.Tensor,
        src_conf: torch.Tensor,
        dst_conf: torch.Tensor,
    ) -> torch.Tensor:
        feature_delta = src_features - dst_features
        coord_delta = src_coords - dst_coords
        return torch.cat(
            [
                src_features,
                dst_features,
                feature_delta,
                coord_delta,
                src_conf,
                dst_conf,
            ],
            dim=-1,
        )
