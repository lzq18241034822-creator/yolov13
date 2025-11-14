# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import csv
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import cv2

# 允许从项目根目录导入 utils
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
from utils.matching import match_detections


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_label_file(path: str, imgsz: int) -> Tuple[List[List[int]], List[List[int]], List[np.ndarray]]:
    """
    解析统一分割标签：
    每行：5 个整数（五头标签） + 多对归一化 polygon 坐标。
    将 polygon 映射到 letterbox 后的 640x640（或 imgsz×imgsz）坐标，输出：
    - gt_boxes: xyxy（letterbox 坐标）
    - gt_ids5: [species, cell_org, shape, flagella, chloroplast]
    - gt_masks: 二值掩膜（letterbox 坐标，uint8）
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # 解析像素维度（原始尺寸）
    W = H = None
    for ln in lines:
        if ln.startswith('# pixel_dimensions_px:'):
            dims = ln.split(':', 1)[1].strip()
            W, H = [int(x) for x in dims.split('x')]
            break
    if W is None or H is None:
        # 默认从文件名或配置缺失时，回退到 640x640
        W, H = imgsz, imgsz

    # letterbox 参数
    r = min(imgsz / float(W), imgsz / float(H))
    nw, nh = int(round(W * r)), int(round(H * r))
    dw = (imgsz - nw) / 2.0
    dh = (imgsz - nh) / 2.0

    gt_boxes, gt_ids5, gt_masks = [], [], []
    for ln in lines:
        if ln.startswith('#'):
            continue
        parts = ln.split()
        if len(parts) < 15:
            # 至少需要 5 类别 + 10 坐标（5 点）
            continue
        ids = [int(parts[i]) for i in range(5)]
        coords = [float(x) for x in parts[5:]]
        pts = []
        for i in range(0, len(coords), 2):
            x = coords[i] * W
            y = coords[i + 1] * H
            x = x * r + dw
            y = y * r + dh
            pts.append([x, y])
        pts = np.array(pts, dtype=np.float32)
        if len(pts) < 3:
            continue
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        gt_boxes.append([x1, y1, x2, y2])
        gt_ids5.append(ids)
        # 栅格化掩膜
        m = np.zeros((imgsz, imgsz), dtype=np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 1)
        gt_masks.append(m)
    return gt_boxes, gt_ids5, gt_masks


def macro_f1(cm: np.ndarray):
    # cm[gt, pred]
    eps = 1e-9
    K = cm.shape[0]
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def save_cm_png(cm: np.ndarray, out_path: str):
    m = cm.astype(np.float32)
    if m.size == 0:
        m = np.zeros((1, 1), dtype=np.float32)
    m = m / (m.max() + 1e-9)
    m = (m * 255.0).astype(np.uint8)
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    h, w = m.shape[:2]
    # 放大便于预览
    m = cv2.resize(m, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, m)


def main():
    ap = argparse.ArgumentParser(description='Stage3 评测与报告')
    ap.add_argument('--cfg', type=str, default='', help='可选：用于加载类别数量或规则')
    ap.add_argument('--data_root', type=str, required=True, help='统一数据集根目录')
    ap.add_argument('--pred_json', type=str, default=os.path.join('runs', 'infer_stage3', 'predictions.json'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--save_dir', type=str, default=os.path.join('runs', 'eval_stage3'))
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--plot', type=int, default=1, help='保存 PNG 图像')
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    # 读取预测
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 类别规模：从数据推断最大 id
    max_ids = [0, 0, 0, 0, 0]
    # 统计规则冲突
    rule_total = 0
    rule_conflict = 0

    # 初始化混淆矩阵（稍后根据最大 id 扩容）
    cms = {
        'species': None,
        'cell_org': None,
        'shape': None,
        'flagella': None,
        'chloroplast': None,
    }

    # 检测计数
    det_tp = 0
    det_fp = 0
    det_fn = 0
    total_pred = 0
    total_gt = 0
    iou_accum = []

    # 参数统计
    param_stats = defaultdict(list)

    # 遍历每张图片
    for rec in preds:
        img_path = rec.get('image_path', '')
        if not img_path:
            continue
        # labels 路径由 image_path 映射
        try:
            base = os.path.basename(img_path)
            stem = os.path.splitext(base)[0]
            split = 'train' if ('\\train\\' in img_path or '/train/' in img_path) else 'val'
            label_path = os.path.join(args.data_root, 'labels', split, f'{stem}.txt')
        except Exception:
            continue

        if not os.path.exists(label_path):
            # 统计参数仍可进行
            pass
        else:
            gt_boxes, gt_ids5, gt_masks = parse_label_file(label_path, args.imgsz)
            # 更新类别规模（包含 GT）
            for ids in gt_ids5:
                for k in range(5):
                    max_ids[k] = max(max_ids[k], int(ids[k]))

            # 预测框
            pred_boxes = []
            pred_ids5 = []
            for det in rec.get('detections', []):
                pb = det.get('bbox_xyxy', None)
                if pb is None:
                    continue
                pred_boxes.append([float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])])
                ids_final = det.get('ids5_final', det.get('ids5_raw', {}))
                # 支持 cell_org 为列表（来自离线细化）；其它保持单值
                co_val = ids_final.get('cell_org', 0)
                if isinstance(co_val, list):
                    co_pred = [int(x) for x in co_val if x is not None]
                else:
                    try:
                        co_pred = int(co_val)
                    except Exception:
                        co_pred = 0
                pred_ids5.append([
                    int(ids_final.get('species', 0)),
                    co_pred,
                    int(ids_final.get('shape', 0)),
                    int(ids_final.get('flagella', 0)),
                    int(ids_final.get('chloroplast', 0)),
                ])

                # 规则统计
                rule = det.get('rule', None)
                if rule is not None:
                    rule_total += 1
                    if not bool(rule.get('passed', True)):
                        rule_conflict += 1

                # 类别规模更新（cell_org 列表取其中最大值）
                for k in range(5):
                    v = pred_ids5[-1][k]
                    if k == 1 and isinstance(v, list):
                        if len(v) > 0:
                            max_ids[k] = max(max_ids[k], max(v))
                    else:
                        try:
                            max_ids[k] = max(max_ids[k], int(v))
                        except Exception:
                            pass

            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)
            # 匹配
            mt, pairs = match_detections(pred_boxes, gt_boxes, iou_thr=args.iou_thr)
            matched_pred = set([i for i, j, _ in pairs])
            matched_gt = set([j for i, j, _ in pairs])
            det_tp += len(pairs)
            det_fp += max(0, len(pred_boxes) - len(matched_pred))
            det_fn += max(0, len(gt_boxes) - len(matched_gt))

            # mIoU（使用 bbox IoU 近似）
            for i, j, iou in pairs:
                iou_accum.append(float(iou))

            # 逐头混淆矩阵
            for i, j, _ in pairs:
                gt = gt_ids5[j]
                pd = pred_ids5[i]
                # 延迟初始化矩阵尺寸
                for head_idx, head_name in enumerate(['species', 'cell_org', 'shape', 'flagella', 'chloroplast']):
                    size = max(max_ids[head_idx] + 1, 1)
                    if cms[head_name] is None or cms[head_name].shape[0] < size:
                        old = cms[head_name]
                        new_cm = np.zeros((size, size), dtype=np.int32)
                        if old is not None:
                            new_cm[: old.shape[0], : old.shape[1]] = old
                        cms[head_name] = new_cm
                    gi = int(gt[head_idx])
                    # cell_org 支持列表，优先与 GT 匹配，否则取首项
                    if head_idx == 1 and isinstance(pd[head_idx], list):
                        plist = pd[head_idx]
                        if gi in plist:
                            pi = gi
                        else:
                            pi = int(plist[0]) if len(plist) > 0 else 0
                    else:
                        pi = int(pd[head_idx])
                    if gi < size and pi < size:
                        cms[head_name][gi, pi] += 1

        # 参数统计（若提供）
        params = rec.get('parameters', {})
        for k, v in params.items():
            try:
                param_stats[k].append(float(v))
            except Exception:
                pass

    # 汇总指标
    det_precision = det_tp / (det_tp + det_fp + 1e-9)
    det_recall = det_tp / (det_tp + det_fn + 1e-9)
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-9)
    miou_bbox = float(np.mean(iou_accum)) if len(iou_accum) else 0.0
    rule_conflict_rate = rule_conflict / float(max(1, rule_total))

    head_metrics = {}
    for head_name, cm in cms.items():
        if cm is None:
            head_metrics[head_name] = {'acc': 0.0, 'macro_f1': 0.0}
            continue
        acc = float(np.trace(cm)) / float(cm.sum() + 1e-9)
        mf1 = macro_f1(cm)
        head_metrics[head_name] = {'acc': acc, 'macro_f1': mf1}

    # 保存混淆矩阵和图
    for head_name, cm in cms.items():
        if cm is None:
            continue
        np.save(os.path.join(args.save_dir, f'cm_{head_name}.npy'), cm)
        if args.plot:
            save_cm_png(cm, os.path.join(args.save_dir, f'cm_{head_name}.png'))

    # 保存参数统计
    with open(os.path.join(args.save_dir, 'params_stats.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['parameter', 'mean', 'std'])
        for k, arr in param_stats.items():
            if len(arr) == 0:
                continue
            w.writerow([k, float(np.mean(arr)), float(np.std(arr))])

    # 保存 summary
    summary = {
        'det_precision': det_precision,
        'det_recall': det_recall,
        'det_f1': det_f1,
        'miou_bbox': miou_bbox,
        'rule_conflict_rate': rule_conflict_rate,
        'heads': head_metrics,
        'counts': {
            'tp': det_tp,
            'fp': det_fp,
            'fn': det_fn,
            'total_pred': total_pred,
            'total_gt': total_gt,
        },
    }
    with open(os.path.join(args.save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Saved to', args.save_dir)


if __name__ == '__main__':
    main()