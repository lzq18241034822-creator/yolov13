# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def kl_div_with_temperature(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor,
                            T: float = 2.0,
                            reduction: str = 'batchmean'):
    """温度 T 的 KL 蒸馏损失。
    student_logits/teacher_logits: (B,C)
    返回值按 Hinton 约定乘以 T^2 进行缩放。
    """
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction=reduction) * (T * T)