import torch
from torch import nn

from criterion.losses.soft_dice import SoftDiceLoss


class SoftDiceCELoss(nn.Module):
    """Weighted SoftDice + cross entropy with multiclass support"""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = SoftDiceLoss(include_background=include_background, smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        if gt.ndim == pred.ndim and gt.shape[1] == 1:
            gt = gt.squeeze(1)

        gt = gt.long()

        dice = self.dice_loss(pred, gt)
        ce = self.ce_loss(pred, gt)
        return self.dice_weight * dice + self.ce_weight * ce
