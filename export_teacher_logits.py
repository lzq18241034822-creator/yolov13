# -*- coding: utf-8 -*-
import os, argparse, numpy as np, torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ruamel.yaml import YAML
try:
    from datasets.roi_multitask_dataset import ROIMultiTaskDataset
except ModuleNotFoundError:
    import importlib.util as _ilu
    _ds_path = os.path.join(ROOT, 'datasets', 'roi_multitask_dataset.py')
    _spec = _ilu.spec_from_file_location('datasets.roi_multitask_dataset', _ds_path)
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    ROIMultiTaskDataset = _mod.ROIMultiTaskDataset

try:
    from models.multitask_heads import MultiHeadClassifier
except ModuleNotFoundError:
    import importlib.util as _ilu2
    _m_path = os.path.join(ROOT, 'models', 'multitask_heads.py')
    _m_spec = _ilu2.spec_from_file_location('models.multitask_heads', _m_path)
    _m_mod = _ilu2.module_from_spec(_m_spec); _m_spec.loader.exec_module(_m_mod)
    MultiHeadClassifier = _m_mod.MultiHeadClassifier


@torch.no_grad()
def export_logits(model, loader, device, out_npz):
    model.eval()
    paths = []
    sp_logits=[]; co_logits=[]; sh_logits=[]; fl_logits=[]; ch_logits=[]
    ids5_list=[]
    for imgs, labels, meta in loader:
        imgs = imgs.to(device)
        outs = model(imgs)
        sp_logits.append(outs['species'].detach().cpu().numpy())
        co_logits.append(outs['cell_org'].detach().cpu().numpy())
        sh_logits.append(outs['shape'].detach().cpu().numpy())
        fl_logits.append(outs['flagella'].detach().cpu().numpy())
        ch_logits.append(outs['chloroplast'].detach().cpu().numpy())
        for m in meta:
            paths.append(m['roi_path'])
            ids5_list.append(m['ids5'])
    np.savez_compressed(out_npz,
        roi_path=np.array(paths),
        species=np.concatenate(sp_logits,0),
        cell_org=np.concatenate(co_logits,0),
        shape=np.concatenate(sh_logits,0),
        flagella=np.concatenate(fl_logits,0),
        chloroplast=np.concatenate(ch_logits,0),
        ids5=np.array(ids5_list, dtype=np.int32))
    print(f"[OK] saved logits npz: {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    default_cfg = os.path.join(os.path.dirname(ROOT), 'µSHM-YOLO', 'yolov13_transformer_unified_v2_1.yaml')
    ap.add_argument("--cfg", required=False, default=default_cfg)
    ap.add_argument("--roi_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roi_size", type=int, default=224)
    ap.add_argument("--out_npz", type=str, default="runs/stage2_teacher/teacher_logits.npz")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--safe_loop", type=int, default=1, help="在 Windows/CPU 上启用无 DataLoader 的安全导出")
    args = ap.parse_args()

    yaml = YAML()
    mean = (0.485,0.456,0.406); std = (0.229,0.224,0.225)
    try:
        with open(args.cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f)
        ds_stats = cfg.get('dataset_stats', None)
        if ds_stats:
            mean = tuple(ds_stats.get('mean', mean))
            std = tuple(ds_stats.get('std', std))
    except Exception:
        pass

    ds = ROIMultiTaskDataset(roi_root=args.roi_root, cfg_path=args.cfg, roi_size=args.roi_size, mean=mean, std=std)
    pin = (args.device.lower().startswith('cuda') and torch.cuda.is_available())
    dl = None if args.safe_loop else DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    device = torch.device(args.device)
    model = MultiHeadClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    if args.safe_loop:
        # 手写安全循环以避免多进程/固定内存导致的崩溃
        model.eval(); paths=[]; sp_logits=[]; co_logits=[]; sh_logits=[]; fl_logits=[]; ch_logits=[]; ids5_list=[]
        bs = max(1, args.batch_size); n = len(ds)
        with torch.no_grad():
            for start in range(0, n, bs):
                end = min(n, start+bs)
                batch=[]; metas=[]
                for j in range(start, end):
                    img_t, labels, meta = ds[j]
                    batch.append(img_t); metas.append(meta)
                imgs = torch.stack(batch, 0).to(device)
                outs = model(imgs)
                sp_logits.append(outs['species'].detach().cpu().numpy())
                co_logits.append(outs['cell_org'].detach().cpu().numpy())
                sh_logits.append(outs['shape'].detach().cpu().numpy())
                fl_logits.append(outs['flagella'].detach().cpu().numpy())
                ch_logits.append(outs['chloroplast'].detach().cpu().numpy())
                for m in metas:
                    paths.append(m['roi_path']); ids5_list.append(m['ids5'])
        np.savez_compressed(args.out_npz,
            roi_path=np.array(paths),
            species=np.concatenate(sp_logits,0),
            cell_org=np.concatenate(co_logits,0),
            shape=np.concatenate(sh_logits,0),
            flagella=np.concatenate(fl_logits,0),
            chloroplast=np.concatenate(ch_logits,0),
            ids5=np.array(ids5_list, dtype=np.int32))
        print(f"[OK] saved logits npz: {args.out_npz}")
    else:
        export_logits(model, dl, device, args.out_npz)


if __name__ == "__main__":
    main()