# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_w = float(bce_weight)
        self.dice_w = float(dice_weight)
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None):
        # logits/targets: (B,C,H,W); valid_mask: (B,C) 或 (B,C,1,1)
        bce = self.bce(logits, targets)  # (B,C,H,W)
        if valid_mask is not None:
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)
            bce = bce * valid_mask
        bce_loss = bce.sum() / (bce.numel() if valid_mask is None else valid_mask.sum().clamp(min=1))

        # Dice (按通道)
        probs = torch.sigmoid(logits)
        inter = (probs * targets)
        union = (probs + targets)
        if valid_mask is not None:
            inter = inter * valid_mask
            union = union * valid_mask
        inter = inter.sum(dim=(0, 2, 3))  # sum over B,H,W
        union = union.sum(dim=(0, 2, 3))
        dice_per_c = (2 * inter + self.eps) / (union + self.eps)
        if valid_mask is not None:
            # 通道掩码 (B,C,1,1) -> (C)，按B聚合已完成，这里只需要按是否存在有效通道归一
            denom = valid_mask.sum(dim=(0, 2, 3)).clamp(min=1)
        else:
            denom = torch.tensor(logits.shape[0], device=logits.device, dtype=inter.dtype)
            denom = denom.repeat(dice_per_c.shape[0])
        dice_mean = (dice_per_c).sum() / denom.sum()
        dice_loss = 1.0 - dice_mean

        return self.bce_w * bce_loss + self.dice_w * dice_loss, {
            'bce': bce_loss.detach(),
            'dice': dice_mean.detach()
        }


def binary_iou(pred_logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None, thresh=0.5):
    # pred/targets: (B,C,H,W)
    preds = (torch.sigmoid(pred_logits) > thresh).float()
    inter = (preds * targets)
    union = (preds + targets).clamp(max=1.0)
    if valid_mask is not None:
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)
        inter = inter * valid_mask
        union = union * valid_mask
    inter = inter.sum()
    union = union.sum().clamp(min=1)
    return (inter / union).detach()