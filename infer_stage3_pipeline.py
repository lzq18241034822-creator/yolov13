# -*- coding: utf-8 -*-
import os, json, glob, argparse
import numpy as np
import cv2
import torch
from ruamel.yaml import YAML

# 确保脚本可导入 µSHM-YOLO 下的包
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 动态导入，增强稳健性
try:
    from models.simple_seg_film import SimpleSegNetFiLM
except ModuleNotFoundError:
    import importlib.util as _ilu
    _p = os.path.join(ROOT, 'models', 'simple_seg_film.py')
    _s = _ilu.spec_from_file_location('models.simple_seg_film', _p)
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    SimpleSegNetFiLM = _m.SimpleSegNetFiLM

try:
    from models.multitask_heads import MultiHeadClassifier
except ModuleNotFoundError:
    import importlib.util as _ilu2
    _p2 = os.path.join(ROOT, 'models', 'multitask_heads.py')
    _s2 = _ilu2.spec_from_file_location('models.multitask_heads', _p2)
    _m2 = _ilu2.module_from_spec(_s2); _s2.loader.exec_module(_m2)
    MultiHeadClassifier = _m2.MultiHeadClassifier

try:
    from stage3.rule_engine import load_rules_from_cfg, check_one, apply_demotion_if_needed
except ModuleNotFoundError:
    import importlib.util as _ilu3
    _p3 = os.path.join(ROOT, 'stage3', 'rule_engine.py')
    _s3 = _ilu3.spec_from_file_location('stage3.rule_engine', _p3)
    _m3 = _ilu3.module_from_spec(_s3); _s3.loader.exec_module(_m3)
    load_rules_from_cfg = _m3.load_rules_from_cfg; check_one = _m3.check_one; apply_demotion_if_needed = _m3.apply_demotion_if_needed

try:
    from stage3.param_calc import compute_image_parameters
except ModuleNotFoundError:
    import importlib.util as _ilu4
    _p4 = os.path.join(ROOT, 'stage3', 'param_calc.py')
    _s4 = _ilu4.spec_from_file_location('stage3.param_calc', _p4)
    _m4 = _ilu4.module_from_spec(_s4); _s4.loader.exec_module(_m4)
    compute_image_parameters = _m4.compute_image_parameters

try:
    from utils.vis_overlay import draw_instance_overlay
except ModuleNotFoundError:
    import importlib.util as _ilu5
    _p5 = os.path.join(ROOT, 'utils', 'vis_overlay.py')
    _s5 = _ilu5.spec_from_file_location('utils.vis_overlay', _p5)
    _m5 = _ilu5.module_from_spec(_s5); _s5.loader.exec_module(_m5)
    draw_instance_overlay = _m5.draw_instance_overlay


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def rasterize_logits_to_instances(logits: torch.Tensor, thr=0.5, min_area=10):
    # logits: (1,C,H,W)
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]  # (C,H,W)
    masks=[]
    for c in range(probs.shape[0]):
        m = (probs[c] > thr).astype(np.uint8)
        if m.sum() >= min_area:
            masks.append(m)
    return masks

def crop_rois_from_masks(img_bgr, masks, expand=1.2, min_size=16):
    H, W = img_bgr.shape[:2]
    crops=[]; boxes=[]
    for m in masks:
        ys, xs = np.where(m>0)
        if len(xs)==0: continue
        x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
        w = x1 - x0; h = y1 - y0
        cx = (x0 + x1)/2; cy = (y0 + y1)/2
        w2 = int(w*expand/2); h2=int(h*expand/2)
        x0e = max(0, int(cx - w2)); x1e = min(W, int(cx + w2))
        y0e = max(0, int(cy - h2)); y1e = min(H, int(cy + h2))
        if (x1e-x0e) < min_size or (y1e-y0e) < min_size:
            continue
        crop = img_bgr[y0e:y1e, x0e:x1e]
        crops.append(crop); boxes.append([x0e,y0e,x1e,y1e])
    return crops, boxes

def normalize_img_rgb(img, mean, std, size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)/255.0
    mean32 = np.array(mean, dtype=np.float32).reshape(1,1,3)
    std32 = np.array(std, dtype=np.float32).reshape(1,1,3)
    img = (img - mean32) / std32
    img = img.transpose(2,0,1)
    return torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # (1,3,H,W)

def build_label_names(cfg, ids5_global):
    # ids5_global: dict with global IDs
    def name_of(gid): return cfg['classes']['names_zh'].get(int(gid), str(gid))
    return {
        'species_name': name_of(ids5_global['species']),
        'cell_org_name': name_of(ids5_global['cell_org']),
        'shape_name': name_of(ids5_global['shape']),
        'flagella_name': name_of(ids5_global['flagella']),
        'chloroplast_name': name_of(ids5_global['chloroplast']),
    }

def main():
    ap = argparse.ArgumentParser()
    # 允许单图输入；cfg 提供默认路径，data_root 在批量模式必填
    ap.add_argument("--cfg", type=str, default=os.path.join(ROOT, 'yolov13_transformer_unified_v2_1.yaml'))
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--image", type=str, default=None, help="单张图片路径，提供则仅分析该图片")
    ap.add_argument("--stage1_weights", required=True, help="tools/reports/best_stage1.pt")
    ap.add_argument("--student_weights", required=True, help="runs/stage2_student/best_student.pt")
    ap.add_argument("--out_dir", type=str, default="runs/infer_stage3")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--max_instances", type=int, default=16)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--split_dir", type=str, default=None, help="如提供，则仅处理 splits/<split>.txt 内样本")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    yaml = YAML()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f)

    # 收集图片：优先单图，其次按 data_root/split
    imgs: list[str] = []
    if args.image:
        if os.path.exists(args.image):
            imgs = [args.image]
        else:
            print(f"[error] 指定的图片不存在：{args.image}")
            return 1
    else:
        if not args.data_root:
            print("[error] 未提供 --data_root，且未指定 --image；无法收集批量图片")
            return 1
        img_dir = os.path.join(args.data_root, 'images', args.split)
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
            imgs.extend(glob.glob(os.path.join(img_dir, ext)))
        imgs = sorted(imgs)
        allowed = None
        if args.split_dir:
            val_list = os.path.join(args.split_dir, f"{args.split}.txt")
            if os.path.exists(val_list):
                with open(val_list, 'r', encoding='utf-8') as f:
                    allowed = set([ln.strip() for ln in f if ln.strip()])
        if allowed is not None:
            imgs = [p for p in imgs if os.path.splitext(os.path.basename(p))[0] in allowed]
    if not imgs:
        print("[warn] 未找到可分析的图片")
        return 0

    device = torch.device(args.device)
    # 加载 Stage1 简易分割模型
    # DCNv3 开关透传：按 YAML
    use_dcnv3 = False
    try:
        use_dcnv3 = bool(cfg['model']['stage1_detection']['backbone']['dcnv3']['enabled'])
    except Exception:
        use_dcnv3 = False
    model_seg = SimpleSegNetFiLM(max_instances=args.max_instances,
                                 base_ch=32,
                                 base_pixel_size=cfg['microscope']['pixel_size_um'],
                                 film_enabled=True,
                                 use_dcnv3=use_dcnv3).to(device)
    ckpt = torch.load(args.stage1_weights, map_location=device)
    model_seg.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model_seg.eval()

    # 加载 Student 多头分类器
    model_cls = MultiHeadClassifier(pretrained=False).to(device)
    model_cls = model_cls.float()
    sd = torch.load(args.student_weights, map_location=device)
    # 仅加载非 backbone 的权重，保留预训练的 float32 backbone，避免 Double 冲突
    if isinstance(sd, dict):
        if 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        sd_partial = {k: (v.float() if isinstance(v, torch.Tensor) else v)
                      for k, v in sd.items() if not k.startswith('backbone.')}
        missing, unexpected = model_cls.load_state_dict(sd_partial, strict=False)
    else:
        model_cls.load_state_dict(sd)
    # 强制全部子模块为 float32
    model_cls.float()
    model_cls.eval()

    rules = load_rules_from_cfg(cfg)
    mean = cfg['dataset_stats']['mean']; std = cfg['dataset_stats']['std']
    px_um = float(cfg['microscope']['pixel_size_um'])

    all_results = []
    for ip in imgs:
        img_bgr = imread_unicode(ip)
        if img_bgr is None:
            print(f"[warn] read fail: {ip}"); continue
        # 分割
        im_res = cv2.resize(img_bgr, (args.imgsz,args.imgsz))
        with torch.no_grad():
            x = torch.from_numpy(cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255.0).unsqueeze(0).to(device)
            logits = model_seg(x, pixel_scale=torch.tensor([px_um], dtype=torch.float32, device=device))
        masks = rasterize_logits_to_instances(logits, thr=args.thr, min_area=20)

        # 计算 ROI & 分类
        crops, boxes = crop_rois_from_masks(im_res, masks, expand=1.2, min_size=16)
        dets=[]; labels_for_vis=[]; rule_infos=[]
        for idx, crop in enumerate(crops):
            inp = normalize_img_rgb(crop, mean, std, size=224).to(device)
            # 兼容部分环境中保存的 Double 权重（CPU 上 conv 需匹配输入类型）
            if next(model_cls.parameters()).dtype == torch.float64:
                inp = inp.double()
            # 诊断 dtype
            try:
                print('[dtype] inp:', inp.dtype, 'conv1:', model_cls.backbone.conv1.weight.dtype,
                      'param0:', next(model_cls.parameters()).dtype)
            except Exception:
                pass
            with torch.no_grad():
                outs = model_cls(inp)
            # 五头预测（取 argmax -> 全局ID）
            sp = int(outs['species'].argmax(dim=1).item())
            co_local = int(outs['cell_org'].argmax(dim=1).item())
            sh_local = int(outs['shape'].argmax(dim=1).item())
            fl_local = int(outs['flagella'].argmax(dim=1).item())
            ch_local = int(outs['chloroplast'].argmax(dim=1).item())
            # 映射到全局ID
            local_to_global = {
                'cell_org': {0:7,1:8,2:9,3:44},
                'shape': {0:12,1:13,2:14,3:15,4:16,5:17,6:18,7:19,8:20,9:21},
                'flagella': {0:22,1:23,2:24,3:25,4:26},
                'chloroplast': {0:27,1:28,2:29,3:30},
            }
            ids5_global = {'species': sp,
                           'cell_org': local_to_global['cell_org'][co_local],
                           'shape': local_to_global['shape'][sh_local],
                           'flagella': local_to_global['flagella'][fl_local],
                           'chloroplast': local_to_global['chloroplast'][ch_local]}
            # 规则检查与降级
            rc = check_one(ids5_global, rules)
            ids5_final = apply_demotion_if_needed(ids5_global, rc)

            # 记录
            dets.append({
                'bbox_xyxy': [int(x) for x in boxes[idx]],
                'ids5_raw': ids5_global,
                'ids5_final': ids5_final,
                'rule': rc
            })
            labels_for_vis.append(build_label_names(cfg, ids5_final))
            rule_infos.append(rc)

        # 计算参数（使用 im_res 尺寸近似，px_um 来源于 YAML）
        params = compute_image_parameters(im_res, masks, pixel_size_um=px_um)

        # 可视化
        vis = draw_instance_overlay(im_res, masks, dets, labels_for_vis, rule_infos)
        out_vis = os.path.join(args.out_dir, f"{os.path.splitext(os.path.basename(ip))[0]}_overlay.jpg")
        cv2.imwrite(out_vis, vis)

        all_results.append({
            'image_path': ip,
            'pixel_size_um': px_um,
            'detections': dets,
            'parameters': params,
            'overlay_path': out_vis
        })

    out_json = os.path.join(args.out_dir, "predictions.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved JSON: {out_json}")
    print(f"[OK] overlays in: {args.out_dir}")
    return 0

if __name__ == "__main__":
    code = main()
    # 兼容作为子进程时返回码
    try:
        import sys as _sys
        _sys.exit(code)
    except Exception:
        pass