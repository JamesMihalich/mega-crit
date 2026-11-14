import surface_distance
import torch
from torch import nn
from surface_distance.metrics import compute_average_surface_distance, compute_surface_distances


class AverageSurfaceDistance(nn.Module):
    """
    AverageSurfaceDistance leverages Google Deepminds surface-distance library

    https://github.com/google-deepmind/surface-distance/blob/master/surface_distance/metrics.py

    ASD returns a tuple, presenting the average distance from gt to pred, and the average
    distance from pred to gt
    """

    def __init__(self, percentage: float, spacing: tuple):
        self.percentage = percentage
        self.spacing = spacing

        if self.percentage < 0 or self.percentage > 1.0:
            raise ValueError(f"Percentage must be between 1 and 0 (currently: {self.percentage})")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple:
        surface_distances = compute_surface_distances(gt, pred, spacing_mm=self.spacing)
        asd = compute_average_surface_distance(surface_distance)

        return asd
