import torch

from caprapose_rt.utils.geometry import spatial_soft_argmax_2d


def test_spatial_soft_argmax_decodes_gaussian_regression_heatmap():
    height = width = 64
    center_x = 21.0
    center_y = 37.0
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    sigma = 2.5
    heatmap = torch.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2.0 * sigma**2))
    coords, confidence = spatial_soft_argmax_2d(heatmap.view(1, 1, height, width))

    assert torch.allclose(coords[0, 0], torch.tensor([center_x, center_y]), atol=0.5)
    assert confidence[0, 0, 0].item() > 0.9
