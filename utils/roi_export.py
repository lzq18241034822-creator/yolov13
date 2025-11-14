# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from PIL import Image


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def export_rois(img_np_hw3, masks_bchw, ids5_list, out_dir, prefix="roi", max_rois=32):
    """
    导出每个实例的ROI裁剪图，并生成/累积 rois.json：
    - img_np_hw3: (H,W,3) uint8
    - masks_bchw: (C,H,W) float32 (单图)
    - ids5_list: [ [species, cell_org, shape, flagella, chloroplast], ... ] 或 None
    - out_dir: 目标目录
    返回：写入的条目数
    """
    _ensure_dir(out_dir)
    H, W = img_np_hw3.shape[:2]
    C = masks_bchw.shape[0]
    cnt = 0

    # 读取已存在的 rois.json（若有），以便累积
    json_path = os.path.join(out_dir, "rois.json")
    try:
        import json
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                rois_meta = json.load(f)
            if not isinstance(rois_meta, list):
                rois_meta = []
        else:
            rois_meta = []
    except Exception:
        rois_meta = []

    for c in range(C):
        if cnt >= max_rois:
            break
        m = masks_bchw[c]
        if m.max() <= 0:
            continue
        ys, xs = np.where(m > 0.5)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad_y = max(2, int(0.05 * (y1 - y0 + 1)))
        pad_x = max(2, int(0.05 * (x1 - x0 + 1)))
        y0 = max(0, y0 - pad_y)
        y1 = min(H - 1, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(W - 1, x1 + pad_x)

        crop = img_np_hw3[y0:y1 + 1, x0:x1 + 1]
        mask_crop = (m[y0:y1 + 1, x0:x1 + 1] > 0.5).astype(np.uint8) * 255

        # 叠色并保存
        overlay = crop.copy()
        overlay[..., 1] = np.maximum(overlay[..., 1], mask_crop)  # G通道增强
        fname = f"{prefix}_c{c}.png"
        ids5 = None
        if ids5_list and c < len(ids5_list):
            ids5 = ids5_list[c]
            # 保留旧命名方式，便于目录直接浏览
            try:
                fname = f"{prefix}_sp{ids5[0]}_co{ids5[1]}_sh{ids5[2]}_fl{ids5[3]}_ch{ids5[4]}_c{c}.png"
            except Exception:
                fname = f"{prefix}_c{c}.png"
        out_path = os.path.join(out_dir, fname)
        Image.fromarray(overlay).save(out_path)

        # 累积 JSON 元数据
        entry = {
            'roi_path': out_path.replace('\\', '/'),
            'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
            'ids5': ids5 if ids5 is not None else None
        }
        rois_meta.append(entry)
        cnt += 1

    # 写回 rois.json
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rois_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return cnt