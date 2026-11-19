import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1 - dice.mean())

def joint_loss(outputs, masks, labels, seg_weight=1.0, cls_weight=1.0):
    seg = DiceBCELoss()(outputs["seg_logits"], masks)
    cls = F.cross_entropy(outputs["cls_logits"], labels)
    return seg_weight * seg + cls_weight * cls, {"seg": seg.detach(), "cls": cls.detach()}
