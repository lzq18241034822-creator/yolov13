"""
Stage2简化版：只训练 species 单头分类器（ResNet18）
适用于50张图的小数据集；从 tools/reports/rois.json 加载 ROI 数据。

使用说明：
- 先确保存在 ROI 与 rois.json（可通过 Stage1 导出或主动学习脚本生成）。
- 默认从 g:/yoloV13/µSHM-YOLO/tools/reports 读取。
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, 'g:/yoloV13/µSHM-YOLO')

# 兼容导入 ROI 数据集（项目内已提供 ROIMultiTaskDataset 类）
try:
    from datasets.roi_multitask_dataset import ROIMultiTaskDataset
except Exception:
    # 动态从文件路径加载
    import importlib.util as _ilu
    _ds_path = os.path.join('g:/yoloV13/µSHM-YOLO', 'datasets', 'roi_multitask_dataset.py')
    _spec = _ilu.spec_from_file_location('datasets.roi_multitask_dataset', _ds_path)
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    ROIMultiTaskDataset = _mod.ROIMultiTaskDataset

from utils.early_stop import EarlyStopping

try:
    from torchvision.models import resnet18
except Exception as e:
    raise ImportError(f"未能导入 torchvision.models.resnet18，请先安装 torchvision：{e}")


class SpeciesClassifier(nn.Module):
    """单头 species 分类器（ResNet18）"""
    def __init__(self, num_species: int = 45, pretrained: bool = True):
        super().__init__()
        self.backbone = resnet18(weights=None if not pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_species)

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for img_t, labels, _ in loader:
        imgs = img_t.to(device)
        species = labels['species'].to(device)

        logits = model(imgs)
        loss = criterion(logits, species)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == species).sum().item()
        total += species.size(0)

    acc = correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for img_t, labels, _ in loader:
        imgs = img_t.to(device)
        species = labels['species'].to(device)
        logits = model(imgs)
        loss = criterion(logits, species)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == species).sum().item()
        total += species.size(0)
    acc = correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


def main():
    ap = argparse.ArgumentParser(description='Stage2 简化：species 单头分类训练')
    ap.add_argument('--cfg', type=str, default='g:/yoloV13/µSHM-YOLO/yolov13_transformer_unified_v2_1.yaml')
    ap.add_argument('--roi_root', type=str, default='g:/yoloV13/µSHM-YOLO/tools/reports')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_dir', type=str, default='g:/yoloV13/runs/stage2_species_only')
    ap.add_argument('--num_species', type=int, default=45, help='输出类别数；默认用全局45类')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 环境安全设置（Windows/CPU）
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    torch.set_num_threads(1)

    # 数据集
    try:
        ds_train = ROIMultiTaskDataset(roi_root=args.roi_root, cfg_path=args.cfg, roi_size=224)
        ds_val = ROIMultiTaskDataset(roi_root=args.roi_root, cfg_path=args.cfg, roi_size=224)
    except FileNotFoundError as e:
        print(f"⚠️ 未找到 rois.json：{e}")
        print("提示：请先运行 Stage1 并导出 ROI，或使用 tools/select_uncertain_rois.py 生成主动学习 ROI。")
        return

    # 简单拆分（8:2）
    num_total = len(ds_train)
    n_val = max(1, int(0.2 * num_total))
    n_train = max(1, num_total - n_val)
    from torch.utils.data import random_split
    ds_train, ds_val = random_split(ds_train, [n_train, n_val])

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"✅ ROI数据集: train={len(ds_train)}, val={len(ds_val)}")

    # 模型
    device = torch.device(args.device)
    model = SpeciesClassifier(num_species=args.num_species, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(patience=10, min_delta=1e-4)

    best_acc = 0.0
    best_path = Path(args.save_dir) / 'best_species.pt'

    print(f"\n🚀 开始训练 species 分类器 (epochs={args.epochs})")
    print("=" * 60)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, loader_train, optimizer, criterion, device)
        va_loss, va_acc = validate(model, loader_val, criterion, device)
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss={tr_loss:.4f} Acc={tr_acc:.3f} | Val Loss={va_loss:.4f} Acc={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), str(best_path))
            print(f"  ✅ 保存 {best_path} (acc={va_acc:.3f})")

        if early_stop.step(va_acc):
            print("⚠️  早停触发")
            break

    print(f"\n✅ 训练完成！最佳 acc={best_acc:.3f}")


if __name__ == '__main__':
    main()