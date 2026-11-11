import torch
import numpy as np

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def calculate_map(pred, target, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []
    for thresh in iou_thresholds:
        iou = iou_score(pred, target)
        ap = 1.0 if iou >= thresh else 0.0
        aps.append(ap)
    return np.mean(aps)

def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    inter = (pred * target).sum()
    return (2 * inter + 1e-7) / (pred.sum() + target.sum() + 1e-7)

def precision_recall(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    return precision.item(), recall.item()
