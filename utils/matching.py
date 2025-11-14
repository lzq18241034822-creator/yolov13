# -*- coding: utf-8 -*-
import numpy as np
import cv2
from typing import List, Tuple

try:
    from scipy.optimize import linear_sum_assignment as _lsa  # type: ignore
except Exception:
    _lsa = None


def iou_xyxy(b1, b2):
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter + 1e-9
    return inter / union


def iou_mask(m1: np.ndarray, m2: np.ndarray):
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum() + 1e-9
    return float(inter) / float(union)


def rasterize_poly_norm(poly_norm: np.ndarray, H: int, W: int):
    poly = (poly_norm * np.array([W, H], dtype=np.float32)).astype(np.int32)
    m = np.zeros((H, W), dtype=np.uint8)
    if len(poly) > 2:
        cv2.fillPoly(m, [poly], 1)
    return m


def _greedy_match(cost: np.ndarray, thr: float):
    # cost = 1 - IoU，形状 [P, G]
    P, G = cost.shape
    matches = [-1] * P
    used_g = set()
    pairs = []
    order = np.argsort(cost, axis=None)  # 从低 cost（高 IoU）到高
    for idx in order:
        i = idx // G
        j = idx % G
        if matches[i] != -1 or j in used_g:
            continue
        iou = 1.0 - float(cost[i, j])
        if iou >= thr:
            matches[i] = j
            used_g.add(j)
            pairs.append((i, j, iou))
    return matches, pairs


def match_detections(pred_boxes: List[List[int]], gt_boxes: List[List[int]], iou_thr=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [-1] * len(pred_boxes), []
    C = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            C[i, j] = 1.0 - float(iou_xyxy(pb, gb))
    if _lsa is not None:
        ri, cj = _lsa(C)
        matches = [-1] * len(pred_boxes)
        pairs = []
        for i, j in zip(ri, cj):
            iou = 1.0 - float(C[i, j])
            if iou >= iou_thr:
                matches[i] = int(j)
                pairs.append((int(i), int(j), iou))
        return matches, pairs
    # 回退贪心匹配
    return _greedy_match(C, iou_thr)