import torch


def prepare_multiclass(pred: torch.Tensor, gt: torch.Tensor):
    if pred.ndim not in (4, 5) or pred.shape[1] < 2 or not pred.is_floating_point():
        raise ValueError("Expected floating 2D or 3D multiclass logits [N, C, ...], C >= 2")

    if gt.ndim == pred.ndim and gt.shape[1] == 1:
        gt = gt.squeeze(1)

    if gt.shape != (pred.shape[0], *pred.shape[2:]):
        raise ValueError("target must contain class indices with shape [N, ...] or [N, 1, ...]")

    if gt.is_floating_point() and (not torch.isfinite(gt).all() or (gt != gt.round()).any()):
        raise ValueError("target must contain integer class indices")

    if (gt < 0).any() or (gt >= pred.shape[1]).any():
        raise ValueError("target class indices must be in [0, C)")

    logits = pred.float() if pred.dtype in (torch.float16, torch.bfloat16) else pred

    return logits, gt.to(device=pred.device, dtype=torch.long)
