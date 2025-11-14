# -*- coding: utf-8 -*-
"""
最小可用的 Stage1 训练/验证循环（分割为主），基于 µSHM-YOLO 统一数据集。

功能：
- 读取 YAML（yolov13_transformer_unified_v2_1.yaml）与数据根目录
- 从 UnifiedSegDataset 取样本，进行 letterbox 与仿射增强（可选）
- 将归一化 polygon 栅格化为掩膜：支持非重叠实例掩膜堆叠或重叠单图标签图
- 使用一个极简的分割网络（SimpleSegNet）进行训练，损失为 BCEWithLogitsLoss
- 每个 epoch 保存若干张训练/验证可视化结果到 tools/reports

说明：这是一个演示型最小循环，不依赖 Ultralytics 训练引擎，便于快速验证掩膜管线与数据兼容性。
"""

import os
import sys
import math
import time
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ruamel.yaml import YAML

# 允许从仓库根导入自定义模块
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from datasets.unified_seg_dataset import UnifiedSegDataset, collate_fn_unified
from utils.aug_poly import letterbox_with_segments, random_affine_with_segments, hflip_with_segments
from models.simple_seg_film import SimpleSegNetFiLM
from utils.losses_seg import BCEDiceLoss, binary_iou
from utils.mask_ops import pack_batch_targets
from utils.roi_export import export_rois
from utils.early_stop import EarlyStopping
from torch.utils.data import WeightedRandomSampler


def segments_to_masks(segs: List[np.ndarray], out_shape: Tuple[int, int], overlap: bool = False):
    """将归一化 polygon 栅格化为掩膜。

    参数：
    - segs: list of (K,2)，归一化坐标
    - out_shape: (H, W)
    - overlap: False → 返回 [N,H,W] 的二值掩膜堆叠；True → 返回 [H,W] 的整图标签图，实例值=1..N

    返回：
    - 如果 overlap=False：np.ndarray (N,H,W) uint8
    - 如果 overlap=True：np.ndarray (H,W) int32，其中 0=背景，1..N=实例编号
    """
    H, W = out_shape
    if not segs:
        return np.zeros((0, H, W), dtype=np.uint8) if not overlap else np.zeros((H, W), dtype=np.int32)

    if overlap:
        label_map = np.zeros((H, W), dtype=np.int32)
        for i, s in enumerate(segs, start=1):
            pts = (s * np.array([W, H], dtype=np.float32)).astype(np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(label_map, [pts], color=int(i))
        return label_map
    else:
        masks = np.zeros((len(segs), H, W), dtype=np.uint8)
        for i, s in enumerate(segs):
            pts = (s * np.array([W, H], dtype=np.float32)).astype(np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(masks[i], [pts], color=1)
        return masks


def xywhn_to_xyxy(labels: np.ndarray, imgsz: int) -> np.ndarray:
    """将归一化 [cls, xc, yc, w, h] 转为像素坐标 [x1,y1,x2,y2]。"""
    if labels.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xc, yc, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    x1 = (xc - w / 2.0) * imgsz
    y1 = (yc - h / 2.0) * imgsz
    x2 = (xc + w / 2.0) * imgsz
    y2 = (yc + h / 2.0) * imgsz
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def vis_overlay_multi(img_bgr: np.ndarray, mask_multi: np.ndarray) -> np.ndarray:
    """将多通道掩膜叠加到 BGR 图像。mask_multi:(C,H,W) 0/1。"""
    overlay = img_bgr.copy()
    if mask_multi.ndim == 3 and mask_multi.shape[0] > 0:
        union = (mask_multi.max(axis=0) > 0).astype(np.uint8)
    else:
        union = (mask_multi > 0).astype(np.uint8)
    color = np.array([0, 255, 0], dtype=np.uint8)  # green
    alpha = 0.25  # 降低叠加强度，避免整屏泛绿
    overlay[union == 1] = (overlay[union == 1] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def vis_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # 保留旧接口用于验证
    return vis_overlay_multi(img_bgr, mask)


def build_loader(data_root: str, split: str, cfg_path: str, imgsz: int, skip_ids: List[int], batch_size: int,
                 id_list_file: str = None, weighted_sample: bool = False):
    ds = UnifiedSegDataset(data_root, split, cfg_path, imgsz, skip_ids, id_list_file=id_list_file)
    if weighted_sample and split == 'train':
        # 计算样本权重：按样本内 species 的逆频率均值
        # 注意：ids5 在 __getitem__ 返回每实例的5元ID，索引0为 species
        species_counts = {}
        species_total = 0
        for ip, lbl_path, px_um in ds.samples:
            # 读取一次标签以统计 species
            objs, _ = ds._read_label(lbl_path)
            for o in objs:
                sp = int(o['ids5'][0])
                species_counts[sp] = species_counts.get(sp, 0) + 1
                species_total += 1
        inv_freq = {sp: (species_total / c) if c > 0 else 0.0 for sp, c in species_counts.items()}
        weights = []
        for ip, lbl_path, px_um in ds.samples:
            objs, _ = ds._read_label(lbl_path)
            if objs:
                w = sum(inv_freq.get(int(o['ids5'][0]), 0.0) for o in objs) / max(1, len(objs))
            else:
                w = 1.0
            weights.append(w)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=collate_fn_unified)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0, collate_fn=collate_fn_unified)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, type=str)
    ap.add_argument('--data_root', required=False, type=str, default=None)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--max_instances', type=int, default=16, help='每图最多实例通道')
    ap.add_argument('--film', type=int, default=1, help='启用FiLM注入')
    ap.add_argument('--use_dcnv3', type=int, default=-1, help='开启DCNv3卷积（-1表示按YAML决定）')
    ap.add_argument('--bce_weight', type=float, default=0.5)
    ap.add_argument('--dice_weight', type=float, default=0.5)
    ap.add_argument('--export_rois', type=int, default=1)
    ap.add_argument('--split_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'splits'))
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--weighted_sample', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    yaml = YAML()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f)
    data_root = args.data_root or cfg['dataset']['path']
    imgsz = args.imgsz or cfg['training']['stage1_training']['imgsz'] if 'training' in cfg else args.imgsz
    skip_ids = cfg['training']['stage1_training']['dataloader']['stage1_class_filter']['skip_species_ids'] \
        if 'training' in cfg else [40, 41, 42, 43]

    # 固定划分列表路径（若存在则启用）
    train_list = os.path.join(args.split_dir, 'train.txt')
    val_list = os.path.join(args.split_dir, 'val.txt')
    # 构建 DataLoader
    train_loader = build_loader(data_root, 'train', args.cfg, imgsz, skip_ids, args.batch_size,
                                id_list_file=train_list if os.path.exists(train_list) else None,
                                weighted_sample=bool(args.weighted_sample))
    try:
        val_loader = build_loader(data_root, 'val', args.cfg, imgsz, skip_ids, args.batch_size,
                                  id_list_file=val_list if os.path.exists(val_list) else None)
    except Exception:
        print("WARN: 未找到 val 集，使用 train 列表作为验证集。")
        val_loader = build_loader(data_root, 'train', args.cfg, imgsz, skip_ids, args.batch_size,
                                  id_list_file=val_list if os.path.exists(val_list) else None)

    device = torch.device(args.device)
    # DCNv3 开关：CLI 优先，默认按 YAML
    yaml_dcn = False
    try:
        yaml_dcn = bool(cfg['model']['stage1_detection']['backbone']['dcnv3']['enabled'])
    except Exception:
        yaml_dcn = False
    use_dcnv3 = bool(yaml_dcn if args.use_dcnv3 == -1 else bool(args.use_dcnv3))

    model = SimpleSegNetFiLM(max_instances=args.max_instances,
                              base_ch=32,
                              base_pixel_size=cfg['microscope']['pixel_size_um'] if 'microscope' in cfg else cfg['dataset']['microscope']['pixel_size_um'],
                              film_enabled=bool(args.film),
                              use_dcnv3=use_dcnv3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)

    save_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(save_dir, exist_ok=True)

    def preprocess_batch(imgs_np, segs_list):
        # imgs_np: (B,H,W,3) uint8; segs_list: list(list of (K,2))
        B = imgs_np.shape[0]
        out_imgs = []
        segs_aug = []
        for bi in range(B):
            img = imgs_np[bi]
            segs = segs_list[bi]
            # 几何一致增强
            img, segs, _, _ = letterbox_with_segments(img, segs, new_shape=(imgsz, imgsz))
            img, segs = random_affine_with_segments(img, segs, degrees=3, translate=0.03, scale=0.1, shear=0)
            if random.random() < 0.5:
                img, segs = hflip_with_segments(img, segs)
            out_imgs.append(img)
            # 归一化segments仍然保持0..1范围
            segs_aug.append([np.asarray(s, dtype=np.float32) for s in segs])

        imgs_t = torch.from_numpy(np.stack(out_imgs, axis=0)).permute(0, 3, 1, 2).contiguous().float() / 255.0
        # 固定C=max_instances的实例掩膜
        targets, valid = pack_batch_targets(segs_aug, imgsz, imgsz, args.max_instances)
        return imgs_t.to(device), targets.to(device), valid.to(device), out_imgs, segs_aug

    def epoch_loop(loader, epoch, is_train=True):
        model.train(is_train)
        total_loss = 0.0
        num_steps = 0
        mean_dice = 0.0
        for batch in loader:
            # 兼容更新后的collate
            imgs_t, labels_t, seg_out, paths, pixel_scales, shapes, ids5_out = batch
            # imgs_t: (B,H,W,3) uint8 from collate; seg_out: list(list of (K,2))
            imgs_t_np = imgs_t.numpy()
            imgs_t_np = imgs_t_np.astype(np.uint8)
            imgs_t_np = imgs_t_np  # (B,H,W,3)
            imgs, targets, valid_mask, imgs_aug, segs_aug = preprocess_batch(imgs_t_np, seg_out)
            pixel_scales_t = (pixel_scales if isinstance(pixel_scales, torch.Tensor) else torch.tensor(pixel_scales, dtype=torch.float32)).to(device)

            if is_train:
                optim.zero_grad()
                pred = model(imgs, pixel_scales_t)  # (B,C,H,W)
                loss, meters = criterion(pred, targets, valid_mask)
                loss.backward()
                optim.step()
            else:
                with torch.no_grad():
                    pred = model(imgs, pixel_scales_t)
                    loss, meters = criterion(pred, targets, valid_mask)

            total_loss += float(loss.detach().cpu().item())
            num_steps += 1
            # 累加 Dice 作为验证指标
            if not is_train:
                mean_dice += float(meters['dice'].cpu().item())

            # 保存可视化：取首张
            with torch.no_grad():
                img0 = imgs_aug[0][:, :, ::-1]  # BGR
                preds_np = (torch.sigmoid(pred[0]).detach().cpu().numpy() > 0.5).astype(np.uint8)  # (C,H,W)
                # 叠加预测（绿）
                vis = vis_overlay_multi(img0, preds_np)
                outp = os.path.join(save_dir, f"train_stage1_epoch{epoch}_{'train' if is_train else 'val'}.jpeg")
                ok, enc = cv2.imencode('.jpeg', vis)
                if ok:
                    enc.tofile(outp)
                # 在图上绘制GT标注轮廓（红）
                vis_gt = vis.copy()
                H, W = vis_gt.shape[:2]
                for s in segs_aug[0]:
                    if s.size == 0:
                        continue
                    pts_px = np.stack([s[:, 0] * W, s[:, 1] * H], axis=1).astype(np.int32)
                    pts_px = pts_px.reshape(-1, 1, 2)
                    cv2.polylines(vis_gt, [pts_px], isClosed=True, color=(0, 0, 255), thickness=2)
                outp_gt = os.path.join(save_dir, f"train_stage1_epoch{epoch}_{'train' if is_train else 'val'}_gt.jpeg")
                ok2, enc2 = cv2.imencode('.jpeg', vis_gt)
                if ok2:
                    enc2.tofile(outp_gt)
                # ROI导出（预测掩膜）
                if args.export_rois and epoch == 1:
                    roidir = os.path.join(save_dir, f"rois_{'train' if is_train else 'val'}_epoch{epoch}")
                    try:
                        export_rois(img0, preds_np, ids5_out[0] if len(ids5_out)>0 else None, roidir, prefix="pred")
                    except Exception:
                        pass

        avg = total_loss / max(1, num_steps)
        if is_train:
            print(f"Train Epoch {epoch}: loss={avg:.4f}")
        else:
            md = mean_dice / max(1, num_steps)
            print(f"Val Epoch {epoch}: loss={avg:.4f}, dice={md:.4f}")

        # 记录损失到CSV
        log_csv = os.path.join(save_dir, 'train_log.csv')
        if not os.path.exists(log_csv):
            with open(log_csv, 'w', encoding='utf-8') as f:
                f.write('epoch,split,loss,dice\n')
        with open(log_csv, 'a', encoding='utf-8') as f:
            if is_train:
                f.write(f"{epoch},train,{avg:.6f},\n")
            else:
                f.write(f"{epoch},val,{avg:.6f},{md:.6f}\n")

    # 训练/验证
    start_epoch = 1
    best_metric = -1e9
    # resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            print(f"Resumed from {args.resume}, next epoch {start_epoch}")
        except Exception:
            print("WARN: resume failed, starting fresh.")
    early = EarlyStopping(patience=args.patience, min_delta=1e-4)

    for ep in range(start_epoch, args.epochs + 1):
        epoch_loop(train_loader, ep, is_train=True)
        # 验证并获取最新 dice（从日志文件读取或直接复算一轮）
        # 直接跑一轮验证以便拿到 md
        # 将验证期间的最后 md 写入作用域变量：简便起见，复跑一次取返回值不改结构
        # 这里复用 epoch_loop 输出到日志即可，再做一次快速评估
        # 简化：调用一次验证并在日志中读取最后一行 dice
        epoch_loop(val_loader, ep, is_train=False)
        # 保存权重
        ckpt = {"model": model.state_dict(), "optim": optim.state_dict(), "epoch": ep}
        torch.save(ckpt, os.path.join(save_dir, "last_stage1.pt"))
        # 读取日志的最后一个 val 行以取 dice
        val_dice = None
        try:
            with open(os.path.join(save_dir, 'train_log.csv'), 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for ln in reversed(lines):
                parts = ln.split(',')
                if len(parts) >= 4 and parts[1] == 'val' and int(parts[0]) == ep:
                    val_dice = float(parts[3]) if parts[3] else None
                    break
        except Exception:
            pass
        if val_dice is None:
            val_dice = -1e9
        if val_dice > best_metric:
            best_metric = val_dice
            torch.save(ckpt, os.path.join(save_dir, "best_stage1.pt"))
        early.step(val_dice)
        if early.should_stop:
            print(f"Early stopping at epoch {ep}. Best dice={best_metric:.4f}")
            break


if __name__ == '__main__':
    main()