import math

import torch
import torch.nn.functional as F
from torch import nn

from .soft_dice import SoftDiceLoss
from .svls import SVLSLoss


class SVLSDiceLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        svls_weight: float = 1.0,
        sigma: float = 1.0,
        kernel_size: int = 3,
        include_background: bool = True,
        smooth: float = 1e-5,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.svls_weight = svls_weight

        self.dice_loss = SoftDiceLoss(include_background=include_background, smooth=smooth)
        self.svls_loss = SVLSLoss(sigma=sigma, kernel_size=kernel_size)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = prediction.new_zeros(())
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice_loss(prediction, target)
        if self.svls_weight:
            loss = loss + self.svls_weight * self.svls_loss(prediction, target)
        return loss
