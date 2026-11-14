import math

import torch
import torch.nn.functional as F
from torch import nn


def svls_kernel(dimensions: int, kernel_size: int, sigma: float) -> torch.Tensor:

    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)

    grids = torch.meshgrid(*([coordinates] * dimensions), indexing="ij")
    distance_squared = sum(grid.square() for grid in grids)

    center = (radius,) * dimensions

    distance_squared[center] = float("inf")
    weights = torch.exp(-0.5 * ((distance_squared - 1) / sigma) / sigma)
    weights *= 0.5 / weights.sum()
    weights[center] = 0.5
    return weights.float().reshape(1, 1, *weights.shape)


class SVLSLoss(nn.Module):
    """
    Spatially Varying Label Smoothing followed by soft-target cross-entropy, directly from the paper!

    https://github.com/mobarakol/SVLS/blob/main/svls.py
    """

    def __init__(self, sigma: float = 1.0, kernel_size: int = 3):
        super().__init__()
        if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 3
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be an odd integer >= 3")
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.register_buffer("kernel_2d", svls_kernel(2, kernel_size, sigma))
        self.register_buffer("kernel_3d", svls_kernel(3, kernel_size, sigma))

    @torch.no_grad()
    def smooth_targets(self, one_hot: torch.Tensor) -> torch.Tensor:

        if one_hot.ndim not in (4, 5) or not one_hot.is_floating_point():
            raise ValueError("one_hot must be a floating 2D or 3D batched class tensor")

        dimensions = one_hot.ndim - 2
        kernel = self.kernel_2d if dimensions == 2 else self.kernel_3d

        kernel = kernel.to(device=one_hot.device, dtype=one_hot.dtype)
        weights = kernel.expand(one_hot.shape[1], 1, *kernel.shape[2:]).contiguous()
        padded = F.pad(one_hot, [self.kernel_size // 2] * (2 * dimensions), mode="replicate")
        conv = F.conv2d if dimensions == 2 else F.conv3d

        # Target probabilities should retain their precision under trainer AMP.
        with torch.autocast(device_type=one_hot.device.type, enabled=False):
            return conv(padded, weights, groups=one_hot.shape[1])

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim not in (4, 5) or prediction.shape[1] < 2:
            raise ValueError("SVLS requires 2D or 3D multiclass logits [N, C, ...] with C >= 2")
        if target.ndim == prediction.ndim and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.shape != (prediction.shape[0], *prediction.shape[2:]):
            raise ValueError("target must contain class indices with shape [N, ...] or [N, 1, ...]")
        logits = (
            prediction.float()
            if prediction.dtype in (torch.float16, torch.bfloat16)
            else prediction
        )
        with torch.no_grad():
            one_hot = F.one_hot(target.long(), num_classes=prediction.shape[1]).movedim(-1, 1)
            soft_target = self.smooth_targets(one_hot.to(logits.dtype))
        return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
