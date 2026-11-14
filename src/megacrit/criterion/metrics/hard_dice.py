import torch
from torch import nn

from .utilities import prepare_multiclass


class HardDice(nn.Module):
    """Argmax Dice score, average over defined classes then cases. Higher is better always better!"""

    validation_only = True
    higher_is_better = True

    def __init__(self, include_background=False):

        super().__init__()

        if not isinstance(include_background, bool):
            raise ValueError("include_background must be a boolean")

        self.include_background = include_background

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, gt: torch.Tensor):

        logits, labels = prepare_multiclass(pred, gt)

        if labels.numel() == 0 or not torch.isfinite(logits).all():
            raise ValueError("Hard Dice requires nonempty inputs and finite logits")

        hard = logits.argmax(1)
        case_scores = []

        for predicted, expected in zip(hard, labels):
            scores = []
            for cid in range(0 if self.include_background else 1, logits.shape[1]):
                p, y = predicted == cid, expected == cid
                denominator = p.sum() + y.sum()
                if denominator > 0:
                    scores.append(2.0 * (p & y).sum() / denominator)
            # Match surface_dice's finite perfect-empty case convention.
            case_scores.append(torch.stack(scores).mean() if scores else logits.new_tensor(1.0))

        return torch.stack(case_scores).mean()
