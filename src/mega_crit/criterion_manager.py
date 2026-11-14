import torch
from torch import nn

from .criterion.losses.soft_dice import SoftDiceLoss
from .criterion.losses.soft_dice_ce import SoftDiceCELoss
from .criterion.losses.nacl import NACLLoss
from .criterion.losses.svls import SVLSLoss
from .criterion.losses.svls_dsc import SVLSDiceLoss
from .criterion.losses.htls import HTLS

from .criterion.metrics.hard_dice import HardDice


CRITERION_REGISTRY = {
    "soft_dice": SoftDiceLoss,
    "soft_dice_dsc": SoftDiceCELoss,
    "nacl": NACLLoss,
    "svls": SVLSLoss,
    "svls_dsc": SVLSDiceLoss,
    "hard_dice": HardDice,
    "htls": HTLS,
}


class Criterion(nn.Module):
    def __init__(self, criterion: str, criterion_kwargs: dict):

        super().__init__()

        self.criterion_key = criterion
        self.criterion_kwargs = criterion_kwargs

        if not CRITERION_REGISTRY[self.criterion_key]:
            raise ValueError(f"Unrecognized criterion: {self.criterion}")

        self.criterion = CRITERION_REGISTRY[self.criterion_key](self.criterion_kwargs)

    def forward(self, gt: torch.Tensor, pred: torch.Tensor):

        if gt.shape != pred.shape:
            raise ValueError("Ground truth and prediction tensors are not the same shape")

        # criterions are always defined with their forward being pred, gt in that order!
        return self.criterion(pred, gt)
