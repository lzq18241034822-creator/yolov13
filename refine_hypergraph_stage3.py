# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2

# 允许从项目根目录导入 utils 与 stage3
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    from stage3.hypergraph_refine import refine_cell_org_by_proximity
except ModuleNotFoundError:
    import importlib.util as _ilu1
    _p1 = os.path.join(ROOT, 'stage3', 'hypergraph_refine.py')
    _s1 = _ilu1.spec_from_file_location('stage3.hypergraph_refine', _p1)
    _m1 = _ilu1.module_from_spec(_s1); _s1.loader.exec_module(_m1)
    refine_cell_org_by_proximity = _m1.refine_cell_org_by_proximity

try:
    from utils.vis_overlay import draw_instance_overlay
except ModuleNotFoundError:
    import importlib.util as _ilu2
    _p2 = os.path.join(ROOT, 'utils', 'vis_overlay.py')
    _s2 = _ilu2.spec_from_file_location('utils.vis_overlay', _p2)
    _m2 = _ilu2.module_from_spec(_s2); _s2.loader.exec_module(_m2)
    draw_instance_overlay = _m2.draw_instance_overlay

try:
    from ruamel.yaml import YAML
except Exception:
    YAML = None


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_names_zh(cfg_path: str) -> Dict[int, str]:
    if not cfg_path or YAML is None or not os.path.exists(cfg_path):
        return {}
    y = YAML()
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = y.load(f)
    names_zh = cfg.get('classes', {}).get('names_zh', {})
    # 键可能为字符串，统一成 int
    out = {}
    for k, v in names_zh.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out


def id_to_name(idval: Any, names_zh: Dict[int, str]) -> str:
    try:
        iid = int(idval)
        return names_zh.get(iid, str(iid))
    except Exception:
        return str(idval)


def rect_mask_from_bbox(b: List[float], H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return m
    m[y1:y2, x1:x2] = 1
    return m


def build_labels_for_vis(detections: List[Dict[str, Any]], names_zh: Dict[int, str], refined_cell_org: List[int]) -> List[Dict[str, Any]]:
    labels = []
    cell_org_text = '、'.join([id_to_name(i, names_zh) for i in refined_cell_org]) if refined_cell_org else ''
    for det in detections:
        ids = det.get('ids5_final') or det.get('ids5_raw') or {}
        sp = ids.get('species', None)
        sh = ids.get('shape', None)
        fl = ids.get('flagella', None)
        ch = ids.get('chloroplast', None)
        labels.append({
            'species_name': id_to_name(sp, names_zh),
            'cell_org_name': cell_org_text or id_to_name(ids.get('cell_org', None), names_zh),
            'shape_name': id_to_name(sh, names_zh),
            'flagella_name': id_to_name(fl, names_zh),
            'chloroplast_name': id_to_name(ch, names_zh),
        })
    return labels


def main():
    ap = argparse.ArgumentParser(description='Stage3 超图（邻近聚团）离线细化')
    ap.add_argument('--pred_json', type=str, default=os.path.join('runs', 'infer_stage3', 'predictions.json'))
    ap.add_argument('--cfg', type=str, default=os.path.join(ROOT, 'yolov13_transformer_unified_v2_1.yaml'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--eps_factor', type=float, default=0.08, help='聚团半径系数，相对 min(H,W)')
    ap.add_argument('--min_size_px', type=int, default=16, help='忽略过小 ROI')
    ap.add_argument('--out_dir', type=str, default=os.path.join('runs', 'infer_stage3_hg'))
    ap.add_argument('--plot', type=int, default=1, help='保存叠加图')
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    names_zh = load_names_zh(args.cfg)

    refined_all = []
    for rec in preds:
        ip = rec.get('image_path')
        img = imread_unicode(ip)
        if img is None:
            # 若原图不可读，降级只在 JSON 内进行，不保存叠加
            H = W = args.imgsz
            im_res = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            im_res = cv2.resize(img, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        H, W = args.imgsz, args.imgsz

        dets = rec.get('detections', [])
        refined_cell_org = refine_cell_org_by_proximity(dets, (H, W), eps_factor=args.eps_factor, min_size_px=args.min_size_px)

        # 写回 refined ids 到每个实例（cell_org 字段置为列表）
        for det in dets:
            ids5 = det.get('ids5_final')
            if not ids5:
                ids5 = det.get('ids5_raw') or {}
                det['ids5_final'] = ids5
            ids5['cell_org'] = refined_cell_org.copy()

        out_rec = {
            'image_path': ip,
            'pixel_size_um': rec.get('pixel_size_um'),
            'detections': dets,
            'parameters': rec.get('parameters', {}),
        }

        # 叠加图（用矩形掩膜近似叠加）
        overlay_path = None
        if int(args.plot) == 1:
            masks = [rect_mask_from_bbox(d.get('bbox_xyxy', [0, 0, 0, 0]), H, W) for d in dets]
            labels_for_vis = build_labels_for_vis(dets, names_zh, refined_cell_org)
            rule_infos = [d.get('rule', {'passed': True}) for d in dets]
            vis = draw_instance_overlay(im_res, masks, dets, labels_for_vis, rule_infos)
            overlay_path = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(ip))[0]}_overlay_hg.jpg")
            ok, enc = cv2.imencode('.jpg', vis)
            if ok:
                enc.tofile(overlay_path)
                out_rec['overlay_path'] = overlay_path

        refined_all.append(out_rec)

    # 输出 refined JSON
    out_json = os.path.join(args.out_dir, 'refined_predictions.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(refined_all, f, ensure_ascii=False, indent=2)
    # 写一个 Vite 客户端占位避免 404 刷屏
    try:
        with open(os.path.join(args.out_dir, '@vite', 'client'), 'w', encoding='utf-8') as f:
            f.write('// placeholder for vite client in static server')
    except Exception:
        pass
    print(f"[OK] saved refined JSON: {out_json}")
    print(f"[OK] overlays in: {args.out_dir}")


if __name__ == '__main__':
    main()