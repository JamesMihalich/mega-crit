import surface_distance
import torch
from torch import nn
from surface_distance.metrics import compute_robust_hausdorff, compute_surface_distances


class HausdorffDistance(nn.Module):
    """
    HausdorffDistance leverages google deepminds surface-distance library

    The result is the max perc distance from gt to pred, or from pred to gt

    https://github.com/google-deepmind/surface-distance/blob/master/surface_distance/metrics.py
    """

    def __init__(self, percentage: float, spacing: tuple):
        self.percentage = percentage
        self.spacing = spacing

        if self.percentage < 0 or self.percentage > 1.0:
            raise ValueError(f"Percentage must be between 1 and 0 (currently: {self.percentage})")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        surface_distances = compute_surface_distances(gt, pred, spacing_mm=self.spacing)
        hausdorff_value = compute_robust_hausdorff(surface_distances, percent=self.percentage * 100)

        return hausdorff_value
