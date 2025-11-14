# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

try:
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    resnet50 = None
    ResNet50_Weights = None


class MultiHeadClassifier(nn.Module):
    """ResNet50 backbone + 5 classification heads for ROI tasks.
    Heads: species(45), cell_org(4), shape(10), flagella(5), chloroplast(4).
    """
    def __init__(self, pretrained: bool = True, num_species: int = 45,
                 num_cell_org: int = 4, num_shape: int = 10,
                 num_flagella: int = 5, num_chl: int = 4):
        super().__init__()
        if resnet50 is None:
            raise RuntimeError("torchvision not available; please install torchvision to use MultiHeadClassifier")
        weights = ResNet50_Weights.IMAGENET1K_V2 if (pretrained and ResNet50_Weights is not None) else None
        self.backbone = resnet50(weights=weights)
        in_feats = self.backbone.fc.in_features
        # Replace FC with identity; we'll use pooled features
        self.backbone.fc = nn.Identity()

        # Projection + heads
        self.proj = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_feats, 1024),
            nn.ReLU(inplace=True),
        )
        self.head_species = nn.Linear(1024, num_species)
        self.head_cell_org = nn.Linear(1024, num_cell_org)
        self.head_shape = nn.Linear(1024, num_shape)
        self.head_flagella = nn.Linear(1024, num_flagella)
        self.head_chl = nn.Linear(1024, num_chl)

    def forward(self, x):
        # x: (B,3,H,W)
        feats = self.backbone(x)  # (B, 2048)
        h = self.proj(feats)
        return {
            'species': self.head_species(h),
            'cell_org': self.head_cell_org(h),
            'shape': self.head_shape(h),
            'flagella': self.head_flagella(h),
            'chloroplast': self.head_chl(h),
        }