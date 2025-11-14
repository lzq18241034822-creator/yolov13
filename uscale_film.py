# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class PixelScaleNormalizer(nn.Module):
    def __init__(self, base_pixel_size=0.0863, method='log_ratio', eps=1e-8):
        super().__init__()
        self.base = float(base_pixel_size)
        self.method = method
        self.eps = eps

    def forward(self, pixel_scale: torch.Tensor):
        # pixel_scale: (B,)
        if self.method == 'log_ratio':
            s = torch.log(pixel_scale.clamp(min=self.eps) / self.base)
        else:
            s = (pixel_scale - self.base) / (self.base + self.eps)
        return s.unsqueeze(-1)  # (B,1)


class FiLM2d(nn.Module):
    def __init__(self, channels: int, base_pixel_size=0.0863, method='log_ratio', hidden=64):
        super().__init__()
        self.norm = PixelScaleNormalizer(base_pixel_size, method=method)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels)
        )
        self.channels = channels

    def forward(self, x: torch.Tensor, pixel_scale: torch.Tensor):
        # x: (B, C, H, W), pixel_scale: (B,)
        s = self.norm(pixel_scale)                # (B,1)
        g = self.mlp(s)                           # (B, 2C)
        gamma, beta = torch.chunk(g, 2, dim=-1)   # (B, C), (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * (1 + gamma) + beta