"""
Stage1训练：YOLOv13分割 + FiLM注入
适配50张图的小数据集策略（强增强与早停）
"""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, 'g:/yoloV13/µSHM-YOLO')

from datasets.unified_seg_dataset import UnifiedSegDataset, collate_fn_unified
from models.yolov13_seg_film import YOLOv13SegFiLM
from utils.losses_seg import BCEDiceLoss
from utils.early_stop import EarlyStopping
from utils.mask_ops import segments_to_masks_batch


def compute_mask_loss(pred_proto, pred_coeffs, pred_cls, gt_masks, gt_labels):
    """
    简化的 YOLOv13 分割损失：
    - 仅使用 P2 层系数（80x80）
    - 随机抽样位置作实例系数（原工程应做 anchor/匹配，这里为跑通最小闭环）
    """
    device = pred_proto.device
    B, num_proto, H_proto, W_proto = pred_proto.shape

    coeff = pred_coeffs[0]
    cls_out = pred_cls[0]

    gt_masks_small = torch.nn.functional.interpolate(
        gt_masks.float(), size=(80, 80), mode='nearest'
    )

    loss_bce_dice = BCEDiceLoss()
    total_loss = 0.0
    num_valid = 0

    for b in range(B):
        valid_mask = gt_labels[b] > 0
        if not valid_mask.any():
            continue
        num_inst = valid_mask.sum().item()
        gt_b = gt_masks_small[b, valid_mask]

        H_c, W_c = coeff.shape[2], coeff.shape[3]
        indices = torch.randint(0, H_c * W_c, (num_inst,), device=device)
        coeff_flat = coeff[b].view(32, -1)
        inst_coeff = coeff_flat[:, indices].T  # (N_inst, 32)

        proto_b = pred_proto[b]  # (32, H_proto, W_proto)
        pred_masks = torch.einsum('nc,chw->nhw', inst_coeff, proto_b)
        pred_masks = torch.sigmoid(pred_masks)
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(0), size=(80, 80), mode='bilinear', align_corners=False
        ).squeeze(0)

        loss = loss_bce_dice(pred_masks, gt_b)
        total_loss += loss
        num_valid += 1

    if num_valid == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / num_valid


def train_one_epoch(model, loader, optimizer, device, max_instances: int = 16):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        imgs = batch['images'].to(device)
        segments = batch['segments']
        labels = batch['labels']
        pixel_size_um = batch.get('pixel_size_um', None)
        if pixel_size_um is not None:
            pixel_size_um = pixel_size_um.to(device)

        B, _, H, W = imgs.shape
        gt_masks, gt_labels_mask = segments_to_masks_batch(
            segments, labels, H, W, device, max_instances=max_instances
        )

        out = model(imgs, pixel_size_um)
        if 'mask' in out and out['proto'] is None:
            pred_mask = out['mask']
            gt_merged = (gt_masks.sum(dim=1, keepdim=True) > 0).float()
            loss_fn = BCEDiceLoss()
            loss = loss_fn(torch.sigmoid(pred_mask), gt_merged)
        else:
            loss = compute_mask_loss(out['proto'], out['coeffs'], out['cls'], gt_masks, gt_labels_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 5 == 0:
            print(f"  Batch [{i+1}/{len(loader)}] Loss: {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, max_instances: int = 16):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        imgs = batch['images'].to(device)
        segments = batch['segments']
        labels = batch['labels']
        pixel_size_um = batch.get('pixel_size_um', None)
        if pixel_size_um is not None:
            pixel_size_um = pixel_size_um.to(device)

        B, _, H, W = imgs.shape
        gt_masks, gt_labels_mask = segments_to_masks_batch(
            segments, labels, H, W, device, max_instances=max_instances
        )

        out = model(imgs, pixel_size_um)
        if 'mask' in out and out['proto'] is None:
            pred_mask = out['mask']
            gt_merged = (gt_masks.sum(dim=1, keepdim=True) > 0).float()
            loss_fn = BCEDiceLoss()
            loss = loss_fn(torch.sigmoid(pred_mask), gt_merged)
        else:
            loss = compute_mask_loss(out['proto'], out['coeffs'], out['cls'], gt_masks, gt_labels_mask)

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description='Stage1: YOLOv13+FiLM 分割训练')
    parser.add_argument('--cfg', type=str, default='g:/yoloV13/µSHM-YOLO/yolov13_transformer_unified_v2_1.yaml')
    parser.add_argument('--data_root', type=str, default='g:/yoloV13/µSHM-YOLO/samples')
    parser.add_argument('--split_dir', type=str, default='g:/yoloV13/µSHM-YOLO/splits')
    parser.add_argument('--out_dir', type=str, default='g:/yoloV13/runs/stage1_yolov13_film')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_instances', type=int, default=16)
    parser.add_argument('--safe_loop', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 环境安全设置（Windows/CPU）
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    torch.set_num_threads(1)

    device = torch.device(args.device)
    model = YOLOv13SegFiLM(num_classes=1, imgsz=args.imgsz)
    model.to(device)

    # 数据集
    train_set = UnifiedSegDataset(args.data_root, split='train', cfg_path=args.cfg, imgsz=args.imgsz)
    val_set = UnifiedSegDataset(args.data_root, split='val', cfg_path=args.cfg, imgsz=args.imgsz)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fn_unified, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fn_unified, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(patience=8, min_delta=1e-3)

    best_loss = float('inf')
    best_path = Path(args.out_dir) / 'best.pt'
    last_path = Path(args.out_dir) / 'last.pt'

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.max_instances)
        val_loss = validate(model, val_loader, device, args.max_instances)
        scheduler.step()

        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={time.time()-t0:.1f}s")

        # 保存 last
        torch.save({'model': model.state_dict(), 'epoch': epoch}, last_path)

        # 保存 best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch}, best_path)
            print(f"  ✅ Saved new best: {best_path}")

        # 早停检查
        if early_stop.step(val_loss):
            print("  ⏹️  Early stopping triggered")
            break

    print(f"\n✅ 训练完成：best={best_path}, last={last_path}")


if __name__ == '__main__':
    main()