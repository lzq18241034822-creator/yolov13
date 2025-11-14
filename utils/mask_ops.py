# -*- coding: utf-8 -*-
import numpy as np
import torch


def rasterize_segments_fixed(segs_norm, H, W, max_instances):
    """
    将归一化polygon列表栅格化到固定通道(C=max_instances)，超过的忽略，未满则零填充。
    返回 (masks: (C,H,W) float32, valid: (C,) float32)
    """
    C = int(max_instances)
    masks = np.zeros((C, H, W), dtype=np.float32)
    valid = np.zeros((C,), dtype=np.float32)

    try:
        # 延迟导入cv2，仅在可用时使用；否则退化到点填充近似
        import cv2
        use_cv = True
    except Exception:
        use_cv = False

    idx = 0
    for poly in segs_norm:
        if idx >= C:
            break
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        pts_px = np.stack([pts[:, 0] * W, pts[:, 1] * H], axis=1).astype(np.float32)
        if use_cv:
            rr = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(rr, [pts_px.astype(np.int32)], 255)
            masks[idx] = rr.astype(np.float32) / 255.0
            valid[idx] = 1.0
        else:
            # 极简退化：将顶点投影为点
            rr = np.zeros((H, W), dtype=np.float32)
            for x, y in pts_px:
                xi = int(np.clip(x, 0, W - 1))
                yi = int(np.clip(y, 0, H - 1))
                rr[yi, xi] = 1.0
            masks[idx] = rr
            valid[idx] = 1.0
        idx += 1

    return masks, valid


def pack_batch_targets(batch_segs_norm_list, H, W, max_instances):
    """将批次的归一化segments打包为张量，返回 targets(B,C,H,W) 与 valid(B,C)。"""
    masks_t = []
    valids_t = []
    for segs_norm in batch_segs_norm_list:
        m, v = rasterize_segments_fixed(segs_norm, H, W, max_instances)
        masks_t.append(torch.from_numpy(m))
        valids_t.append(torch.from_numpy(v))
    targets = torch.stack(masks_t, 0)
    valid = torch.stack(valids_t, 0)
    return targets, valid