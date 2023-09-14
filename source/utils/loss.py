import torch
from torch.nn import functional as F


def content_loss(outf, targetf):
    """Content loss."""
    assert len(outf.shape) == 4
    assert len(targetf.shape) == 4
    assert outf.size(0) == targetf.size(0)

    N, c, x, y = outf.shape

    return torch.sum((outf - targetf) ** 2) / (N * c * x * y)


def seg_loss_kd(pred_logits, true_logits, temp):
    """Knowledge-distillation (KD) segmentation loss.

    Args:
        pred_logits: predicted logits
        true_logits: true (teacher) logits
        temp: distillation temperature for softmax
    """
    prev_target_probs = F.softmax(true_logits / temp, dim=1)
    pred_log_probs = F.log_softmax(pred_logits / temp, dim=1)
    kd_loss = torch.mean(-torch.sum(prev_target_probs * pred_log_probs, dim=1))
    return kd_loss
