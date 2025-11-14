# -*- coding: utf-8 -*-
import os, sys, cv2, numpy as np
from ruamel.yaml import YAML

# 让脚本可从仓库根导入 datasets/ 与 utils/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from datasets.unified_seg_dataset import UnifiedSegDataset
from utils.aug_poly import letterbox_with_segments, random_affine_with_segments, hflip_with_segments

def draw_poly_box(img, segs, color=(0,255,0)):
    h, w = img.shape[:2]
    for s in segs:
        pts = (s * np.array([w, h])).astype(int)
        for i in range(len(pts)):
            p1, p2 = tuple(pts[i]), tuple(pts[(i+1)%len(pts)])
            cv2.line(img, p1, p2, color, 2)
        x0,x1, y0,y1 = s[:,0].min(), s[:,0].max(), s[:,1].min(), s[:,1].max()
        cv2.rectangle(img, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), (0,255,255), 2)
    return img

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--data_root", required=False, type=str, default=None,
                    help="优先使用此数据根目录覆盖 YAML 的 dataset.path")
    args = ap.parse_args()

    yaml = YAML()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f)
    data_root = args.data_root or cfg['dataset']['path']
    imgsz = cfg['training']['stage1_training']['imgsz']
    skip_ids = cfg['training']['stage1_training']['dataloader']['stage1_class_filter']['skip_species_ids']

    ds = UnifiedSegDataset(data_root, 'train', args.cfg, imgsz, skip_ids)
    # 直接取前两张样本，避免依赖 torch DataLoader
    num = min(2, len(ds))
    # 预览输出目录固定到 tools/reports
    out_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(out_dir, exist_ok=True)

    for bi in range(num):
        item = ds[bi]
        img = item['im']
        segs = item['segments']
        img, segs, _, _ = letterbox_with_segments(img, segs, new_shape=(imgsz, imgsz))
        img, segs = random_affine_with_segments(img, segs, degrees=5, translate=0.05, scale=0.2, shear=0)
        if np.random.rand()<0.5:
            img, segs = hflip_with_segments(img, segs)
        vis = draw_poly_box(img.copy(), segs)
        fn = os.path.basename(item['path'])
        outp = os.path.join(out_dir, f"preview_{fn}")
        ok, enc = cv2.imencode('.jpeg', vis)
        if ok:
            enc.tofile(outp)
            print(f"saved {outp}")
        else:
            print(f"failed to encode image for {outp}")