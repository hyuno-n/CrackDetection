import torch
from torch import Tensor

def calculate_precision_recall(pred: Tensor, true: Tensor, epsilon: float = 1e-6):
    assert pred.shape == true.shape, f'Pred and True tensors must have the same shape, but got {pred.shape} and {true.shape}'
    
    tp = torch.sum(pred * true, dim=(-1, -2))  # 픽셀 단위에서 TP 계산
    fp = torch.sum(pred * (1 - true), dim=(-1, -2))  # False Positive 계산
    fn = torch.sum((1 - pred) * true, dim=(-1, -2))  # False Negative 계산
    
    # 정밀도(Precision)와 재현율(Recall) 계산
    precision = tp / (tp + fp + epsilon)  # 1e-8은 나누기 0 방지
    recall = tp / (tp + fn + epsilon)
    
    return precision, recall

def multiclass_calculate(pred: Tensor, true: Tensor):
    return calculate_precision_recall(pred.flatten(0, 1), true.flatten(0, 1))

def calculate_f1_score(precision: Tensor, recall: Tensor, epsilon: float = 1e-6):
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score.mean()