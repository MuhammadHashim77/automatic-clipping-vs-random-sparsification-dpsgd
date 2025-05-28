# Dice coefficient for binary segmentation tasks.
# This function computes the Dice coefficient, a measure of overlap between two binary masks.
import torch

def dice(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    inter = (pred & target.bool()).sum(dim=(2, 3)).float()
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)).float()
    return ((2 * inter + eps) / (union + eps)).mean()
