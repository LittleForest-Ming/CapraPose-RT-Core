import torch

from caprapose_rt.constants import KEYPOINT_NAMES
from caprapose_rt.models.losses import TopologyConsistentStructuralLoss


def test_structural_loss_respects_zero_weights() -> None:
    loss_module = TopologyConsistentStructuralLoss(
        bone_ratio_weight=0.0,
        symmetry_weight=0.0,
        angle_weight=0.0,
    )
    pred_coords = torch.rand(2, len(KEYPOINT_NAMES), 2)
    gt_coords = torch.rand(2, len(KEYPOINT_NAMES), 2)
    visibility = torch.ones(2, len(KEYPOINT_NAMES))

    losses = loss_module(pred_coords, gt_coords, visibility)
    assert losses["bone_ratio"].item() == 0.0
    assert losses["symmetry"].item() == 0.0
    assert losses["angle"].item() == 0.0
    assert losses["total"].item() == 0.0
