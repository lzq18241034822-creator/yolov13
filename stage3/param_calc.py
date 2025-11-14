# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import math

def mask_area_perimeter(mask_bin: np.ndarray) -> Tuple[float, float]:
    # mask_bin: (H,W) {0,1}
    area_px = float(mask_bin.sum())
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim_px = 0.0
    for cnt in contours:
        perim_px += cv2.arcLength(cnt, True)
    return area_px, perim_px

def major_minor_axis(mask_bin: np.ndarray) -> Tuple[float, float, float]:
    # 返回: major_px, minor_px, angle_deg（长轴方向角）
    cnts, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        rect = cv2.minAreaRect(cnt)
        (w,h) = rect[1]
        a = float(max(w,h)); b = float(min(w,h)); ang = float(rect[2])
        return a, b, ang
    ell = cv2.fitEllipse(cnt)
    (cx,cy), (MA,ma), angle = ell  # OpenCV返回MA>=ma
    return float(MA), float(ma), float(angle)

def compactness(area_px: float, perim_px: float) -> float:
    if perim_px <= 1e-6:
        return 0.0
    return float((4 * math.pi * area_px) / (perim_px ** 2))

def ellipse_iou(mask_bin: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0.0
    ell = cv2.fitEllipse(cnt)
    ell_mask = np.zeros_like(mask_bin, dtype=np.uint8)
    cv2.ellipse(ell_mask, ell, 1, -1)
    inter = int((ell_mask & mask_bin).sum())
    union = int((ell_mask | mask_bin).sum())
    if union == 0:
        return 0.0
    return float(inter / union)

def green_channel_index(image_bgr: np.ndarray, mask_bin: np.ndarray) -> float:
    # 绿通量简单指标：G / 255 在mask内像素平均
    g = image_bgr[...,1].astype(np.float32) / 255.0
    m = mask_bin.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float(g[m].mean())

def nearest_neighbor_cv(centroids: List[Tuple[float,float]]) -> float:
    # 最近邻距离的变异系数 CV
    if len(centroids) < 2:
        return 0.0
    C = np.array(centroids, dtype=np.float32)
    dists=[]
    for i in range(len(C)):
        others = C[np.arange(len(C)) != i]
        if len(others) == 0:
            continue
        di = np.sqrt(((C[i] - others)**2).sum(axis=1))
        if len(di) == 0:
            continue
        dists.append(di.min())
    if not dists:
        return 0.0
    d = np.array(dists, dtype=np.float32)
    if d.mean() <= 1e-6:
        return 0.0
    return float(d.std(ddof=1) / d.mean())

def area_cv(areas: List[float]) -> float:
    if len(areas) < 2:
        return 0.0
    a = np.array(areas, dtype=np.float64)
    if a.mean() <= 1e-6:
        return 0.0
    return float(a.std(ddof=1) / a.mean())

def nematic_order_parameter(angles_deg: List[float]) -> float:
    # S ≈ 平均 |cos(2θ)|
    if not angles_deg:
        return 0.0
    th = np.deg2rad(np.array(angles_deg, dtype=np.float32))
    s = np.abs(np.cos(2*th)).mean()
    return float(s)

def compute_image_parameters(image_bgr: np.ndarray,
                             inst_masks: List[np.ndarray],
                             pixel_size_um: float) -> Dict[str,float]:
    # 实例级指标
    areas_px=[]; perims_px=[]; majors_px=[]; minors_px=[]; angles=[]
    centroids=[]
    for m in inst_masks:
        area_px, perim_px = mask_area_perimeter(m)
        A, B, ang = major_minor_axis(m)
        areas_px.append(area_px); perims_px.append(perim_px)
        majors_px.append(A); minors_px.append(B); angles.append(ang)
        ys, xs = np.where(m>0)
        if len(xs)>0:
            centroids.append((float(xs.mean()), float(ys.mean())))
    # 单位换算
    px = float(pixel_size_um)
    areas_um2 = (np.array(areas_px) * (px**2)).tolist()
    perims_um = (np.array(perims_px) * px).tolist()
    majors_um = (np.array(majors_px) * px).tolist()
    minors_um = (np.array(minors_px) * px).tolist()

    # 图像物理面积（mm^2）
    H, W = image_bgr.shape[:2]
    area_mm2 = (W*px/1000.0) * (H*px/1000.0)
    density = len(inst_masks) / area_mm2 if area_mm2 > 0 else 0.0

    # 其它
    comp_vals = [compactness(a,p) if p>0 else 0.0 for a,p in zip(areas_px, perims_px)]
    ell_iou = [ellipse_iou(m) for m in inst_masks]
    green_idx = np.mean([green_channel_index(image_bgr, m) for m in inst_masks]) if inst_masks else 0.0

    params = {
        'cell_density': float(density),
        'aspect_ratio': float(np.mean([(ma/(mi+1e-6)) if mi>0 else 0.0 for ma,mi in zip(majors_px, minors_px)]) if inst_masks else 0.0),
        'boundary_complexity': float(np.mean([(p**2)/(4*math.pi*a + 1e-6) if a>0 else 0.0 for a,p in zip(areas_px, perims_px)]) if inst_masks else 0.0),
        'fractal_dimension': 0.0,  # 可选实现 box-counting，先置0
        'compactness': float(np.mean(comp_vals) if comp_vals else 0.0),
        'ellipse_similarity': float(np.mean(ell_iou) if ell_iou else 0.0),
        'chlorophyll_content': float(green_idx),
        'aggregation_degree': float(nearest_neighbor_cv(centroids)),
        'size_uniformity': float(area_cv(areas_um2)),
        'orientation_orderliness': float(nematic_order_parameter(angles)),
    }
    return params