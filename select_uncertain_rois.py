# -*- coding: utf-8 -*-
"""
主动学习挑样：筛选模型预测最不确定的样本（接近0.5的概率），并导出其ROI。

用法示例：
python µSHM-YOLO/tools/select_uncertain_rois.py --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml \
    --weights µSHM-YOLO/tools/reports/best_stage1.pt --split train --k 32
"""

import os
import sys
import math
from typing import List, Tuple

import numpy as np
import torch
from ruamel.yaml import YAML

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from datasets.unified_seg_dataset import UnifiedSegDataset, collate_fn_unified
from utils.aug_poly import letterbox_with_segments
from utils.mask_ops import pack_batch_targets
from models.simple_seg_film import SimpleSegNetFiLM
from utils.roi_export import export_rois


def preprocess_infer(imgs_np, segs_list, imgsz, device, max_instances):
    B = imgs_np.shape[0]
    out_imgs = []
    segs_out = []
    for bi in range(B):
        img = imgs_np[bi]
        segs = segs_list[bi]
        img, segs, _, _ = letterbox_with_segments(img, segs, new_shape=(imgsz, imgsz))
        out_imgs.append(img)
        segs_out.append([np.asarray(s, dtype=np.float32) for s in segs])
    imgs_t = torch.from_numpy(np.stack(out_imgs, axis=0)).permute(0,3,1,2).contiguous().float() / 255.0
    targets, valid = pack_batch_targets(segs_out, imgsz, imgsz, max_instances)
    return imgs_t.to(device), targets.to(device), valid.to(device), out_imgs, segs_out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, type=str)
    ap.add_argument('--data_root', required=False, type=str, default=None)
    ap.add_argument('--weights', type=str, default=os.path.join(os.path.dirname(__file__), 'reports', 'best_stage1.pt'))
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--k', type=int, default=32, help='挑选Top-K不确定样本')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--max_instances', type=int, default=16)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--split_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'splits'))
    ap.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'reports', 'active_rois_topk'))
    args = ap.parse_args()

    yaml = YAML()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f)
    data_root = args.data_root or cfg['dataset']['path']
    imgsz = args.imgsz or cfg['training']['stage1_training']['imgsz'] if 'training' in cfg else args.imgsz
    device = torch.device(args.device)

    # 构建数据集（固定划分列表如果存在则使用）
    id_list_file = None
    list_path = os.path.join(args.split_dir, f'{args.split}.txt')
    if os.path.exists(list_path):
        id_list_file = list_path
    skip_ids = cfg['training']['stage1_training']['dataloader']['stage1_class_filter']['skip_species_ids'] \
        if 'training' in cfg else [40,41,42,43]
    ds = UnifiedSegDataset(data_root, args.split, args.cfg, imgsz, skip_ids, id_list_file=id_list_file)

    # 模型与权重
    # DCNv3 开关透传：按 YAML
    use_dcnv3 = False
    try:
        use_dcnv3 = bool(cfg['model']['stage1_detection']['backbone']['dcnv3']['enabled'])
    except Exception:
        use_dcnv3 = False
    model = SimpleSegNetFiLM(max_instances=args.max_instances,
                             base_ch=32,
                             base_pixel_size=cfg['microscope']['pixel_size_um'] if 'microscope' in cfg else cfg['dataset']['microscope']['pixel_size_um'],
                             film_enabled=True,
                             use_dcnv3=use_dcnv3).to(device)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded weights: {args.weights}")
    else:
        print("WARN: 未找到权重，使用随机初始化模型。")
    model.eval()

    # 遍历数据集计算不确定性
    scores = []  # (uncert, img_aug, prob_np, ids5_list)
    for i in range(len(ds)):
        b = ds[i]
        # collate 一致输入
        imgs_t = np.expand_dims(b['im'], 0)
        segs_list = [b['segments']]
        ids5_list = b.get('ids5', [])
        imgs, targets, valid, imgs_aug, segs_aug = preprocess_infer(imgs_t, segs_list, imgsz, device, args.max_instances)
        with torch.no_grad():
            pixel_scales_t = torch.tensor([b['pixel_size_um']], dtype=torch.float32, device=device)
            logits = model(imgs, pixel_scales_t)
            prob = torch.sigmoid(logits)
            # 不确定性：越接近0.5越大
            uncert = (0.5 - torch.abs(prob - 0.5)).mean().item()
            scores.append((uncert, imgs_aug[0], prob[0].detach().cpu().numpy(), ids5_list))

    # 选择Top-K不确定样本
    scores.sort(key=lambda x: x[0], reverse=True)
    topk = scores[:args.k]
    os.makedirs(args.out_dir, exist_ok=True)
    for j, (sc, img0, prob_np, ids5_list) in enumerate(topk):
        # 阈值化导出ROI
        masks = (prob_np > 0.5).astype(np.uint8)
        prefix = f"uncertain_s{sc:.4f}_idx{j}"
        try:
            export_rois(img0, masks, ids5_list, args.out_dir, prefix=prefix, max_rois=32)
        except Exception as e:
            print(f"ROI导出失败: {e}")


if __name__ == '__main__':
    main()