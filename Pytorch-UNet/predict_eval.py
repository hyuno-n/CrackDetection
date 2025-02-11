import torch
import torch.nn.functional as F
import argparse
from PIL import Image
import numpy as np
from utils.dice_score import dice_coeff, multiclass_dice_coeff
from utils.calculate_iou import calculate_iou
from utils.F1_score import calculate_f1_score, calculate_precision_recall

def evaluate_metrics(mask_pred, mask_true, n_classes):
    """
    Calculate F1 score, Dice score, and IoU for predicted and true masks.

    Args:
        mask_pred (torch.Tensor): Predicted mask, shape [B, H, W] or [B, C, H, W].
        mask_true (torch.Tensor): True mask, shape [B, H, W] or [B, C, H, W].
        n_classes (int): Number of classes (1 for binary, >1 for multiclass).

    Returns:
        dict: Dictionary containing F1 score, Dice score, and IoU.
    """
    if n_classes == 1:
        # Binary segmentation
        assert mask_true.min() >= 0 and mask_true.max() <= 1, "True mask indices should be in [0, 1]"
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        dice = dice_coeff(mask_pred, mask_true, reduce_batch_first=False).item()
        iou = calculate_iou(mask_pred, mask_true)

        precision, recall = calculate_precision_recall(mask_pred, mask_true)
        f1 = calculate_f1_score(precision, recall).item()
    else:
        # Multiclass segmentation
        assert mask_true.min() >= 0 and mask_true.max() < n_classes, "True mask indices should be in [0, n_classes]"
        
        # Convert masks to one-hot format
        mask_true = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred, n_classes).permute(0, 3, 1, 2).float()

        # Compute metrics ignoring the background (class 0)
        dice = multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False).item()
        iou = calculate_iou(mask_pred[:, 1:], mask_true[:, 1:])

        precision, recall = calculate_precision_recall(mask_pred[:, 1:], mask_true[:, 1:])
        f1 = calculate_f1_score(precision, recall).item()

    return {
        'f1_score': f1,
        'dice_score': dice,
        'iou': iou
    }

def load_mask(file_path):
    """
    Load a mask file as a PyTorch tensor.
    Supports .pt, .jpg, and .png files.

    Args:
        file_path (str): Path to the mask file.

    Returns:
        torch.Tensor: Loaded mask as a PyTorch tensor.
    """
    if file_path.endswith('.pt'):
        return torch.load(file_path)
    elif file_path.endswith(('.jpg', '.png')):
        # Open image and convert to array
        image = Image.open(file_path)
        mask = np.array(image)

        # Ensure the mask is either boolean, integer, or float
        if mask.dtype == np.bool_:
            # Convert boolean mask to torch.bool
            return torch.tensor(mask, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        elif mask.dtype in [np.int32, np.int64, np.uint8, np.float32, np.float64]:
            # Convert numeric mask to torch.float32
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # If the mask has 0 and 255 values, convert 255 to 1
            mask = torch.where(mask == 255, torch.tensor(1.0), mask)
            return mask  # Add batch dimension
        else:
            raise ValueError(f"Unexpected data type in mask: {mask.dtype}")

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for predicted and true masks.")
    parser.add_argument("-p", "--predict", required=True, help="Path to the predicted mask file")
    parser.add_argument("-m", "--mask", required=True, help="Path to the true mask file")
    parser.add_argument("-c", "--classes", type=int, required=True, help="Number of classes (1 for binary, >1 for multiclass)")
    args = parser.parse_args()

    # Load masks
    mask_pred = load_mask(args.predict)
    mask_true = load_mask(args.mask)

    # Evaluate metrics
    metrics = evaluate_metrics(mask_pred, mask_true, args.classes)

    # Print results
    print("Evaluation Metrics:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Dice Score: {metrics['dice_score']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")

if __name__ == "__main__":
    main()
