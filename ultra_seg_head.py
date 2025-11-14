# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvBNActLite(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.cv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(True)

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))


class Proto(nn.Module):
    """Minimal Proto head producing K prototype maps at input resolution."""

    def __init__(self, c_in: int, c_mid: int = 64, num_protos: int = 32):
        super().__init__()
        self.seq = nn.Sequential(
            ConvBNActLite(c_in, c_mid, 3, 1, 1),
            ConvBNActLite(c_mid, c_mid, 3, 1, 1),
            ConvBNActLite(c_mid, c_mid, 3, 1, 1),
        )
        self.out = nn.Conv2d(c_mid, num_protos, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        p = self.out(x)  # (B, K, H, W)
        return p


class UltraSegHead(nn.Module):
    """
    Minimal YOLOv8-style segmentation head:
    - Generates K prototype maps with Proto
    - Predicts coefficients per instance slot (N slots)
    - Assembles masks: M[n] = sum_k coeff[n,k] * Proto[k]

    Note: This head ignores detection/cropping for simplicity.
    """

    def __init__(self, in_channels: int, num_instances: int = 16, num_protos: int = 32):
        super().__init__()
        self.num_instances = int(num_instances)
        self.num_protos = int(num_protos)

        self.proto = Proto(in_channels, c_mid=64, num_protos=num_protos)
        # Coefficients head: produce (B, N, K) via 1x1 conv + spatial mean
        self.cv_coeff = nn.Conv2d(in_channels, num_instances * num_protos, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prototypes at input resolution
        p = self.proto(x)  # (B, K, H, W)
        B, K, H, W = p.shape
        # Coeffs via 1x1 conv then spatial average
        c = self.cv_coeff(x)  # (B, N*K, H, W)
        c = c.mean(dim=(2, 3))  # (B, N*K)
        c = c.view(B, self.num_instances, self.num_protos)  # (B, N, K)

        # Assemble masks: einsum over proto channels
        # m[b, n, h, w] = sum_k c[b,n,k] * p[b,k,h,w]
        m = torch.einsum('bnk,bkhw->bnhw', c, p)
        return m


class FiLMUNetUltraSeg(nn.Module):
    """
    Compose SimpleSegNetFiLM backbone+decoder features with UltraSegHead.
    Expects the backbone to expose forward_feats(x, pixel_scale) returning the
    last decoder feature before its out_head.
    """

    def __init__(self, backbone, in_channels: int, num_instances: int = 16, num_protos: int = 32):
        super().__init__()
        self.backbone = backbone
        self.seg_head = UltraSegHead(in_channels, num_instances=num_instances, num_protos=num_protos)

    def forward(self, x: torch.Tensor, pixel_scale: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_feats(x, pixel_scale)  # (B, C, H, W)
        logits = self.seg_head(feat)  # (B, N, H, W)
        return logits
