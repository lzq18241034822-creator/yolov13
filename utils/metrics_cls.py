# -*- coding: utf-8 -*-
import numpy as np
import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy."""
    pred = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    return float(correct) / max(1, target.numel())


@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    pred = logits.argmax(dim=1).cpu().numpy()
    targ = target.cpu().numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, targ):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


@torch.no_grad()
def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    cm = confusion_matrix(logits, target, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))