import torch
import torch.nn.functional as F
from torch import nn


class SoftDiceLoss(nn.Module):
    """1 - Dice on softmaxxed logits, lower here is better always better."""

    differentiable = True

    def __init__(
        self,
        include_background: bool = True,
        smooth: float = 1e-5,
        sigmoid: bool = False,
    ):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        # Accumulate large voxel sums in float32 under mixed precision.
        logits = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred

        if self.sigmoid:
            probabilities = logits.sigmoid()
            gt = gt.to(dtype=probabilities.dtype)
        else:
            if gt.ndim == pred.ndim and gt.shape[1] == 1:
                gt = gt.squeeze(1)

            probabilities = logits.softmax(dim=1)

            gt = F.one_hot(gt.long(), num_classes=pred.shape[1]).movedim(-1, 1)
            gt = gt.to(dtype=probabilities.dtype)

            if not self.include_background:
                probabilities = probabilities[:, 1:]
                gt = gt[:, 1:]

        spatial_dims = tuple(range(2, pred.ndim))
        tp = (probabilities * gt).sum(dim=spatial_dims)
        fp = (probabilities * (1 - gt)).sum(dim=spatial_dims)
        fn = ((1 - probabilities) * gt).sum(dim=spatial_dims)
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return (1 - dice).mean()
