# -*- coding: utf-8 -*-
"""
最小实现：Stage1 训练脚本（文字版12任务）
- 使用仓库现有 FiLM2d 与简易分割网络 SimpleSegNetFiLM（含DCNv3占位回退）
- 数据：读取 samples/images/{train,val} + labels-ultra/{train,val}，支持 splits/{train,val}.txt
- 增强：polygon-safe letterbox/affine/hflip
- 目标：跑通1个epoch无错误、透传 pixel_scale 并保存 best 权重（占位指标）

说明：
- 这是最小可运行版本，满足 12.md 的颈部FiLM + DCN占位 + forward透传 pixel_scale + 训练脚本要求。
- 若后续切换到 YOLOv13 真分割头（proto+coeff），可替换 model/criterion 部分为你仓库的真实实现。
"""

import os
import sys
import argparse
import time
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 环境安全设置，避免 Windows/OpenMP 原生冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ULTRALYTICS_IGNORE_WALL_ERRORS", "1")

# OpenCV 降线程，避免与OMP冲突
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    cv2 = None

def _load_module(name: str, file_path: str):
    """从任意文件路径动态导入为模块，避免包名包含特殊字符导致的 import 失败。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {file_path} 加载模块 {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

# 计算仓库根路径
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 动态加载仓库内模块（避免 'µSHM-YOLO' 包名不合法）
_models_dir = os.path.join(REPO_ROOT, 'models')
_utils_dir = os.path.join(REPO_ROOT, 'utils')
msf = _load_module('models_simple_seg_film', os.path.join(_models_dir, 'simple_seg_film.py'))
aug = _load_module('utils_aug_poly', os.path.join(_utils_dir, 'aug_poly.py'))
mop = _load_module('utils_mask_ops', os.path.join(_utils_dir, 'mask_ops.py'))
los = _load_module('utils_losses_seg', os.path.join(_utils_dir, 'losses_seg.py'))
ultra_head_mod = _load_module('models_ultra_seg_head', os.path.join(_models_dir, 'ultra_seg_head.py'))

SimpleSegNetFiLM = msf.SimpleSegNetFiLM
FiLMUNetUltraSeg = ultra_head_mod.FiLMUNetUltraSeg
letterbox_with_segments = aug.letterbox_with_segments
random_affine_with_segments = aug.random_affine_with_segments
hflip_with_segments = aug.hflip_with_segments
pack_batch_targets = mop.pack_batch_targets
BCEDiceLoss = los.BCEDiceLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_yaml_pixel_size(cfg_path: str, default_um: float = 0.0863) -> float:
    """从 YAML 里读取 microscope.pixel_size_um（若不存在则回退默认值）。"""
    try:
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        um = (
            cfg.get('microscope', {}).get('pixel_size_um', None)
            or cfg.get('model', {}).get('pixel_scale_conditioning', {}).get('base_pixel_size_um', None)
        )
        return float(um) if um is not None else float(default_um)
    except Exception:
        return float(default_um)


def read_segments_txt(txt_path: str) -> List[np.ndarray]:
    """YOLO-style segments txt：每行格式 'cls x1 y1 x2 y2 ...'，坐标归一化到[0,1]。
    返回：List[np.ndarray]，每个数组形状(K,2)。
    """
    segs = []
    if not os.path.isfile(txt_path):
        return segs
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                # 跳过 cls
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    # 坐标数不成对，跳过
                    continue
                arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
                segs.append(arr)
    except Exception:
        # 读取异常则返回目前解析到的片段
        pass
    return segs


class MinimalSegDataset(Dataset):
    """最小分割数据集：
    - 根目录：samples
    - 图像：samples/images/{split}/{id}.png
    - 标签：samples/labels-ultra/{split}/{id}.txt （polygon 列表）
    - 支持 splits/{split}.txt 作为固定划分（每行一个id）
    返回：img_u8(BGR)、segments(list[np.ndarray])、pixel_scale(float)、image_id(str)
    """

    def __init__(self, samples_root: str, split: str, split_ids_file: str | None = None):
        super().__init__()
        self.root = samples_root
        self.split = split
        self.img_dir = os.path.join(self.root, 'images', split)
        self.lab_dir = os.path.join(self.root, 'labels-ultra', split)

        # id 列表
        if split_ids_file and os.path.isfile(split_ids_file):
            with open(split_ids_file, 'r', encoding='utf-8') as f:
                ids = [x.strip() for x in f if x.strip()]
        else:
            ids = []
            for name in os.listdir(self.img_dir):
                if name.lower().endswith('.png'):
                    ids.append(os.path.splitext(name)[0])
        self.ids = sorted(ids, key=lambda x: int(x))
        # 过滤不存在的图像ID，避免 splits 与文件夹不一致导致报错
        self.ids = [iid for iid in self.ids if os.path.isfile(os.path.join(self.img_dir, f"{iid}.png"))]

        # 兼容异常文件名（如含空格的'26 .txt'）
        self._label_alt = set(os.listdir(self.lab_dir)) if os.path.isdir(self.lab_dir) else set()

    def __len__(self):
        return len(self.ids)

    def _find_label_path(self, iid: str) -> str:
        p = os.path.join(self.lab_dir, f"{iid}.txt")
        if os.path.isfile(p):
            return p
        # 兼容类似 '26 .txt'
        alt = f"{iid} .txt"
        if alt in self._label_alt:
            return os.path.join(self.lab_dir, alt)
        return p  # 默认返回标准路径，可能不存在

    def __getitem__(self, idx: int):
        iid = self.ids[idx]
        ip = os.path.join(self.img_dir, f"{iid}.png")
        lp = self._find_label_path(iid)

        if cv2 is None:
            raise RuntimeError("OpenCV 未安装，无法加载图像。请安装 opencv-python。")
        # 兼容 Windows 非ASCII路径：使用 imdecode 读取
        try:
            data = np.fromfile(ip, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            raise FileNotFoundError(f"图像不存在或无法读取: {ip}")
        segs = read_segments_txt(lp)
        return img, segs, iid


def collate_minibatch(batch):
    # batch: List[(img_u8, segs_list, id)]
    imgs, segs, ids = [], [], []
    for i, (im, s, iid) in enumerate(batch):
        imgs.append(im)
        segs.append(s)
        ids.append(iid)
    return np.stack(imgs, 0), segs, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=os.path.join(REPO_ROOT, 'yolov13_transformer_unified_v2_1.yaml'))
    ap.add_argument('--samples_root', default=os.path.join(REPO_ROOT, 'samples'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--split_dir', type=str, default=os.path.join(REPO_ROOT, 'splits'))
    ap.add_argument('--use_dcnv3', action='store_true', help='打开DCNv3（若不可用会自动回退）')
    ap.add_argument('--use_ultra_head', action='store_true', help='使用最小YOLOv8风格分割头（proto+coeff）')
    ap.add_argument('--save_last', action='store_true', help='每个epoch后保存last.pt')
    ap.add_argument('--eval_best', action='store_true', help='训练后载入best.pt并在val集评估一次')
    args = ap.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    pixel_um = parse_yaml_pixel_size(args.cfg, default_um=0.0863)

    # 数据集与加载器
    train_ids_file = os.path.join(args.split_dir, 'train.txt') if args.split_dir else None
    val_ids_file = os.path.join(args.split_dir, 'val.txt') if args.split_dir else None
    ds_train = MinimalSegDataset(args.samples_root, 'train', train_ids_file)
    ds_val = MinimalSegDataset(args.samples_root, 'val', val_ids_file)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=False, collate_fn=collate_minibatch)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=False, collate_fn=collate_minibatch)

    # 模型（FiLM 注入 + DCNv3占位回退）
    base_ch = 32
    if args.use_ultra_head:
        backbone = SimpleSegNetFiLM(max_instances=16, base_ch=base_ch, base_pixel_size=pixel_um,
                                    film_enabled=True, use_dcnv3=args.use_dcnv3).to(device)
        model = FiLMUNetUltraSeg(backbone, in_channels=base_ch, num_instances=16, num_protos=32).to(device)
    else:
        model = SimpleSegNetFiLM(max_instances=16, base_ch=base_ch, base_pixel_size=pixel_um,
                                 film_enabled=True, use_dcnv3=args.use_dcnv3).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    save_dir = os.path.join('runs', 'stage1_yolov13')
    os.makedirs(save_dir, exist_ok=True)
    best_score = -1.0

    H = W = int(args.imgsz)

    def run_epoch(dataloader: DataLoader, train: bool):
        model.train(train)
        losses = []
        dices = []
        for imgs_u8, segs_list, ids in dataloader:
            # 逐样本 polygon-safe 预处理与增强
            batch_imgs = []
            batch_segs_norm = []
            for bi in range(len(imgs_u8)):
                img = imgs_u8[bi]
                segs = segs_list[bi]
                img, segs, _, _ = letterbox_with_segments(img, segs, new_shape=(H, W))
                # 轻量增强（安全范围）
                img, segs = random_affine_with_segments(img, segs, degrees=10, translate=0.1, scale=0.30, shear=0)
                if np.random.rand() < 0.5:
                    img, segs = hflip_with_segments(img, segs)
                batch_imgs.append(img)
                batch_segs_norm.append(segs)

            # 栅格化目标 (B,C,H,W) 与有效通道掩码(B,C)
            targets, valid = pack_batch_targets(batch_segs_norm, H, W, max_instances=16)

            # 图像组装到张量 (B,3,H,W) [BGR->RGB]
            images = []
            for im in batch_imgs:
                x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0)
                images.append(x)
            images = torch.stack(images, 0).to(device)
            targets = targets.to(device)
            valid = valid.to(device)

            # 像素尺度 (B,) 常量（若后续从TXT头部读取则替换）
            px = torch.full((images.size(0),), float(pixel_um), device=device)

            # 前向与损失
            logits = model(images, pixel_scale=px)
            loss, stats = criterion(logits, targets, valid)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            losses.append(float(loss.detach().cpu().numpy()))
            dices.append(float(stats['dice'].cpu().numpy()))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_dice = float(np.mean(dices)) if dices else 0.0
        return mean_loss, mean_dice

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice = run_epoch(dl_train, train=True)
        va_loss, va_dice = run_epoch(dl_val, train=False)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f} dice={tr_dice:.4f} | val_loss={va_loss:.4f} dice={va_dice:.4f}  ({dt:.1f}s)")

        # 以 val dice 作为占位评分（越大越好）
        score = va_dice
        if score > best_score:
            best_score = score
            save_path = os.path.join(save_dir, 'best.pt')
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'pixel_um': pixel_um}, save_path)
            print(f"  [best] saved {save_path}")
        if args.save_last:
            last_path = os.path.join(save_dir, 'last.pt')
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'pixel_um': pixel_um}, last_path)
            print(f"  [last] saved {last_path}")

    # 训练结束后可选评估 best.pt
    if args.eval_best:
        best_path = os.path.join(save_dir, 'best.pt')
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            print(f"[eval_best] loaded epoch={ckpt.get('epoch')} pixel_um={ckpt.get('pixel_um')}")
            va_loss, va_dice = run_epoch(dl_val, train=False)
            print(f"[eval_best] val_loss={va_loss:.4f} dice={va_dice:.4f}")
        else:
            print(f"[eval_best] best.pt not found at {best_path}")


if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        print(f"[SystemExit] code={getattr(e, 'code', None)}")
        raise
    except Exception as e:
        import traceback
        print("[ERROR]", e)
        traceback.print_exc()