# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
try:
    # 正常包内相对导入（在作为包加载时生效）
    from .uscale_film import FiLM2d
except Exception:
    # 动态导入：在作为独立模块加载时生效
    import os, sys, importlib.util
    _THIS_DIR = os.path.dirname(__file__)
    _USCALE_PATH = os.path.join(_THIS_DIR, 'uscale_film.py')
    spec = importlib.util.spec_from_file_location('models_uscale_film', _USCALE_PATH)
    if spec is None or spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    sys.modules['models_uscale_film'] = _mod
    FiLM2d = _mod.FiLM2d


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, use_dcnv3: bool = False):
        super().__init__()
        # 可选 DCNv3，若不可用则回退
        self._has_dcnv3 = False
        self._use_dcnv3 = bool(use_dcnv3)
        if self._use_dcnv3:
            try:
                from dcnv3 import DCNv3 as DCNConv  # 占位示例，实际库名可能不同
                self.cv = DCNConv(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
                self._has_dcnv3 = True
            except Exception:
                print("[warn] DCNv3 not available; fallback to Conv2d.")
                self.cv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        else:
            self.cv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(True)

    def forward(self, x):
        return self.act(self.bn(self.cv(x)))


class SimpleSegNetFiLM(nn.Module):
    """轻量U-Net风格 + FiLM 注入，输出固定 max_instances 通道。"""

    def __init__(self, max_instances: int = 16, base_ch: int = 32, base_pixel_size: float = 0.0863,
                 film_enabled: bool = True, use_dcnv3: bool = False):
        super().__init__()
        self.max_instances = int(max_instances)
        self.film_enabled = bool(film_enabled)

        # 编码器（可选 DCNv3 回退）
        self.enc1 = nn.Sequential(ConvBNAct(3, base_ch, use_dcnv3=use_dcnv3),
                                  ConvBNAct(base_ch, base_ch, use_dcnv3=use_dcnv3))
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(ConvBNAct(base_ch, base_ch * 2, use_dcnv3=use_dcnv3),
                                  ConvBNAct(base_ch * 2, base_ch * 2, use_dcnv3=use_dcnv3))
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(ConvBNAct(base_ch * 2, base_ch * 4, use_dcnv3=use_dcnv3),
                                  ConvBNAct(base_ch * 4, base_ch * 4, use_dcnv3=use_dcnv3))

        self.film1 = FiLM2d(base_ch, base_pixel_size)
        self.film2 = FiLM2d(base_ch * 2, base_pixel_size)
        self.film3 = FiLM2d(base_ch * 4, base_pixel_size)

        # 解码器
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = nn.Sequential(ConvBNAct(base_ch * 4, base_ch * 2, use_dcnv3=use_dcnv3),
                                  ConvBNAct(base_ch * 2, base_ch * 2, use_dcnv3=use_dcnv3))
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(ConvBNAct(base_ch * 2, base_ch, use_dcnv3=use_dcnv3),
                                  ConvBNAct(base_ch, base_ch, use_dcnv3=use_dcnv3))

        # 固定输出为 max_instances 通道
        self.out_head = nn.Conv2d(base_ch, self.max_instances, 1, 1, 0)

    def forward(self, x, pixel_scale: torch.Tensor):
        e1 = self.enc1(x)
        if self.film_enabled:
            e1 = self.film1(e1, pixel_scale)

        x = self.down1(e1)
        e2 = self.enc2(x)
        if self.film_enabled:
            e2 = self.film2(e2, pixel_scale)

        x = self.down2(e2)
        e3 = self.enc3(x)
        if self.film_enabled:
            e3 = self.film3(e3, pixel_scale)

        x = self.up2(e3)
        x = torch.cat([x, e2], 1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, e1], 1)
        x = self.dec1(x)
        logits = self.out_head(x)  # (B, C=max_instances, H, W)
        return logits

    def forward_feats(self, x, pixel_scale: torch.Tensor):
        """返回 out_head 之前的最后解码特征，用于外部分割头。"""
        e1 = self.enc1(x)
        if self.film_enabled:
            e1 = self.film1(e1, pixel_scale)

        x = self.down1(e1)
        e2 = self.enc2(x)
        if self.film_enabled:
            e2 = self.film2(e2, pixel_scale)

        x = self.down2(e2)
        e3 = self.enc3(x)
        if self.film_enabled:
            e3 = self.film3(e3, pixel_scale)

        x = self.up2(e3)
        x = torch.cat([x, e2], 1)
        x = self.dec2(x)
        x = self.up1(x)
        x = torch.cat([x, e1], 1)
        x = self.dec1(x)
        return x
