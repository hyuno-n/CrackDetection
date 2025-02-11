import torch
from torch import Tensor

def calculate_iou(true_mask: Tensor, pred_mask: Tensor) -> float:
    """
    Calculate Intersection over Union (IoU) for binary masks.

    Args:
        true_mask (Tensor): Ground truth binary mask (torch.Tensor).
        pred_mask (Tensor): Predicted binary mask (torch.Tensor).

    Returns:
        float: IoU score.
    """
    assert true_mask.shape == pred_mask.shape, "Shapes of true_mask and pred_mask must match"

    # Intersection: logical AND between true_mask and pred_mask
    intersection = torch.logical_and(true_mask, pred_mask).sum().item()

    # Union: logical OR between true_mask and pred_mask
    union = torch.logical_or(true_mask, pred_mask).sum().item()

    if union == 0:
        return 0.0  # Avoid division by zero

    iou = intersection / union
    return iou
