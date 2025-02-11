import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.calculate_iou import calculate_iou
from utils.F1_score import calculate_f1_score, calculate_precision_recall, multiclass_calculate

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou = 0
    f1_score_total = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
        
            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.6).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true.float(), reduce_batch_first=False).item()
                iou += calculate_iou(mask_pred.squeeze(1), mask_true.float())
                # Compute precision, recall, and F1 score
                precision, recall = calculate_precision_recall(mask_pred.squeeze(1), mask_true.float())
                f1 = calculate_f1_score(precision, recall)
                f1_score_total += f1.item()
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False).item()
                iou += calculate_iou(mask_pred[:, 1:], mask_true[:, 1:])
                # Compute precision, recall, and F1 score for each class
                precision, recall = multiclass_calculate(mask_pred[:, 1:], mask_true[:, 1:])
                f1 = calculate_f1_score(precision, recall)
                f1_score_total += f1.item()
    net.train()
    return float(dice_score / max(num_val_batches, 1)), float(iou / max(num_val_batches, 1)), float(f1_score_total / max(num_val_batches, 1))
