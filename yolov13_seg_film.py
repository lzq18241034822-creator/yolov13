"""
YOLOv13分割模型 + FiLM注入
基于 iMoonLab/yolov13 的 Segment 架构（若不可用则回退到 SimpleSegNetFiLM）。
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn


# 动态导入 yolov13-main 的模块
YOLOV13_PATH = Path('g:/yoloV13/yolov13-main')
if str(YOLOV13_PATH) not in sys.path:
    sys.path.insert(0, str(YOLOV13_PATH))

try:
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, SPPF
    # 这里不直接使用 Segment 类，而是自定义最小原型+系数头
    YOLOV13_AVAILABLE = True
except Exception as e:
    print(f"⚠️  无法导入YOLOv13模块: {e}")
    print(f"   请确认 {YOLOV13_PATH} 存在且包含 ultralytics/nn/")
    YOLOV13_AVAILABLE = False
    Conv = None
    C2f = None
    SPPF = None


sys.path.insert(0, 'g:/yoloV13/µSHM-YOLO')
from models.uscale_film import PixelScaleNormalizer, FiLM2d


class YOLOv13SegFiLM(nn.Module):
    """
    YOLOv13分割 + FiLM像素尺度注入（最小实现）
    - Backbone: YOLOv13-nano 级简化，FiLM 注入于 P2/P3/P4
    - Neck: 简化 PAN
    - Head: 最小 proto + coeff + cls 多层输出
    """
    def __init__(self, num_classes: int = 1, imgsz: int = 640, pixel_size_um_default: float = 0.0863):
        super().__init__()
        self.num_classes = num_classes
        self.imgsz = imgsz
        self.scale_norm = PixelScaleNormalizer(default_pixel_size=pixel_size_um_default)

        if not YOLOV13_AVAILABLE:
            print("⚠️  YOLOv13模块不可用，使用简易分割网络回退模式")
            from models.simple_seg_film import SimpleSegNetFiLM
            self.model = SimpleSegNetFiLM(num_classes=num_classes)
            self.is_fallback = True
            return

        self.is_fallback = False

        # ========== Backbone (YOLOv13-nano simplified) ==========
        self.stem = Conv(3, 16, k=3, s=2)  # 640 → 320
        self.stage1 = nn.Sequential(Conv(16, 32, k=3, s=2), C2f(32, 32, n=1))  # 320 → 160
        self.stage2 = nn.Sequential(Conv(32, 64, k=3, s=2), C2f(64, 64, n=2))  # 160 → 80
        self.film_p2 = FiLM2d(64)
        self.stage3 = nn.Sequential(Conv(64, 128, k=3, s=2), C2f(128, 128, n=2))  # 80 → 40
        self.film_p3 = FiLM2d(128)
        self.stage4 = nn.Sequential(Conv(128, 256, k=3, s=2), C2f(256, 256, n=1), SPPF(256, 256, k=5))  # 40 → 20
        self.film_p4 = FiLM2d(256)

        # ========== Neck (简化 PAN) ==========
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p3 = C2f(256 + 128, 128, n=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p2 = C2f(128 + 64, 64, n=1)
        self.down1 = Conv(64, 64, k=3, s=2)
        self.c3_n3 = C2f(64 + 128, 128, n=1)
        self.down2 = Conv(128, 128, k=3, s=2)
        self.c3_n4 = C2f(128 + 256, 256, n=1)

        # ========== Segmentation Head ==========
        # 原型（在 P2 层上采样生成 32 个原型到 320x320）
        self.proto = nn.Sequential(
            Conv(64, 64, k=3),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 80 → 160
            Conv(64, 32, k=3),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 160 → 320
            Conv(32, 32, k=3)
        )

        # 系数与分类（P2/P3/P4）
        self.coeff_p2 = nn.Conv2d(64, 32, 1)
        self.coeff_p3 = nn.Conv2d(128, 32, 1)
        self.coeff_p4 = nn.Conv2d(256, 32, 1)

        self.cls_p2 = nn.Conv2d(64, num_classes, 1)
        self.cls_p3 = nn.Conv2d(128, num_classes, 1)
        self.cls_p4 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor, pixel_size_um: torch.Tensor | None = None):
        if getattr(self, 'is_fallback', False):
            out = self.model(x, pixel_size_um)
            return {'mask': out, 'proto': None, 'coeffs': [], 'cls': []}

        if pixel_size_um is None:
            pixel_size_um = torch.full((x.size(0),), self.scale_norm.default_pixel_size, device=x.device)
        scale_cond = self.scale_norm(pixel_size_um)

        # Backbone
        x = self.stem(x)
        p1 = self.stage1(x)
        p2 = self.stage2(p1)
        p2 = self.film_p2(p2, scale_cond)
        p3 = self.stage3(p2)
        p3 = self.film_p3(p3, scale_cond)
        p4 = self.stage4(p3)
        p4 = self.film_p4(p4, scale_cond)

        # Neck
        x_up = self.up1(p4)
        x_up = torch.cat([x_up, p3], dim=1)
        n3 = self.c3_p3(x_up)
        x_up = self.up2(n3)
        x_up = torch.cat([x_up, p2], dim=1)
        n2 = self.c3_p2(x_up)
        x_down = self.down1(n2)
        x_down = torch.cat([x_down, n3], dim=1)
        n3_out = self.c3_n3(x_down)
        x_down = self.down2(n3_out)
        x_down = torch.cat([x_down, p4], dim=1)
        n4_out = self.c3_n4(x_down)

        # Head
        proto = self.proto(n2)  # (B, 32, 320, 320)
        coeff_p2 = self.coeff_p2(n2)
        coeff_p3 = self.coeff_p3(n3_out)
        coeff_p4 = self.coeff_p4(n4_out)
        cls_p2 = self.cls_p2(n2)
        cls_p3 = self.cls_p3(n3_out)
        cls_p4 = self.cls_p4(n4_out)

        return {
            'proto': proto,
            'coeffs': [coeff_p2, coeff_p3, coeff_p4],
            'cls': [cls_p2, cls_p3, cls_p4]
        }


if __name__ == '__main__':
    # 简单自测
    model = YOLOv13SegFiLM(num_classes=1, imgsz=640)
    x = torch.randn(2, 3, 640, 640)
    pixel_size = torch.tensor([0.0863, 0.1])
    out = model(x, pixel_size)
    if not getattr(model, 'is_fallback', False):
        print("✅ YOLOv13-Seg-FiLM 测试通过")
        print(f"   proto: {out['proto'].shape}")
        print(f"   coeffs: {[c.shape for c in out['coeffs']]}")
        print(f"   cls: {[c.shape for c in out['cls']]}")
    else:
        print("✅ 降级模式测试通过")
        print(f"   mask: {out['mask'].shape}")