# -*- coding: utf-8 -*-
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from ruamel.yaml import YAML

# 让脚本可从仓库根导入 datasets/ 与 models/ 与 utils/
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
    _m = _m_spec.loader.load_module() if hasattr(_m_spec.loader, 'load_module') else None
    if _m is None:
        _m_module = _ilu2.module_from_spec(_m_spec); _m_spec.loader.exec_module(_m_module)
        MultiHeadClassifier = _m_module.MultiHeadClassifier
    else:
        MultiHeadClassifier = _m.MultiHeadClassifier

try:
    from utils.metrics_cls import accuracy, macro_f1, confusion_matrix
except ModuleNotFoundError:
    import importlib.util as _ilu3
    _u_path = os.path.join(ROOT, 'utils', 'metrics_cls.py')
    _u_spec = _ilu3.spec_from_file_location('utils.metrics_cls', _u_path)
    _u_mod = _ilu3.module_from_spec(_u_spec); _u_spec.loader.exec_module(_u_mod)
    accuracy = _u_mod.accuracy; macro_f1 = _u_mod.macro_f1; confusion_matrix = _u_mod.confusion_matrix


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta
        self.best=None; self.count=0; self.should_stop=False
    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric; self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop=True


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__(); self.eps=eps; self.logsoftmax=nn.LogSoftmax(dim=1)
    def forward(self, logits, target):
        n = logits.size(1)
        logp = self.logsoftmax(logits)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.eps / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * logp, dim=1))


def plot_cm_png(cm: np.ndarray, labels: list, out_path: str, normalize=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    M = cm.astype(np.float32)
    if normalize:
        row_sum = M.sum(axis=1, keepdims=True) + 1e-9
        M = M / row_sum
    im = plt.imshow(M, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_species_weights(subset):
    # subset 可能是 torch.utils.data.Subset
    from collections import Counter
    cnt = Counter()
    for i in range(len(subset)):
        _, labels, _ = subset[i]
        cnt[int(labels['species'])] += 1
    total = sum(cnt.values())
    class_w = {k: total / (v + 1e-6) for k, v in cnt.items()}
    weights = []
    for i in range(len(subset)):
        _, labels, _ = subset[i]
        weights.append(class_w[int(labels['species'])])
    return torch.DoubleTensor(weights)


def train_one_epoch(model, loader, optim, device, criterion, log_every=50):
    model.train()
    ls_meter=[]
    for it, (imgs, labels, meta) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        sp = labels['species'].to(device)
        co = labels['cell_org'].to(device)
        sh = labels['shape'].to(device)
        fl = labels['flagella'].to(device)
        ch = labels['chloroplast'].to(device)

        outs = model(imgs)
        ce = criterion
        loss = (ce(outs['species'], sp) +
                ce(outs['cell_org'], co) +
                ce(outs['shape'], sh) +
                ce(outs['flagella'], fl) +
                ce(outs['chloroplast'], ch))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        ls_meter.append(loss.item())
        if (it+1) % log_every == 0:
            print(f"  iter {it+1}/{len(loader)} loss={np.mean(ls_meter):.4f}")
    return float(np.mean(ls_meter)) if ls_meter else 0.0


@torch.no_grad()
def evaluate(model, loader, device, out_dir=None):
    model.eval()
    heads = { 'species': 45, 'cell_org': 4, 'shape': 10, 'flagella': 5, 'chloroplast': 4 }
    stats = {k: {'acc':0,'f1':0,'cm':np.zeros((C,C),dtype=np.int64)} for k, C in heads.items()}
    total = 0
    for imgs, labels, meta in loader:
        imgs = imgs.to(device)
        sp = labels['species'].to(device)
        co = labels['cell_org'].to(device)
        sh = labels['shape'].to(device)
        fl = labels['flagella'].to(device)
        ch = labels['chloroplast'].to(device)
        outs = model(imgs)
        for k, targ, C in [('species',sp,45),('cell_org',co,4),('shape',sh,10),('flagella',fl,5),('chloroplast',ch,4)]:
            acc = accuracy(outs[k], targ)
            f1  = macro_f1(outs[k], targ, C)
            cm  = confusion_matrix(outs[k], targ, C)
            stats[k]['acc'] += acc * imgs.size(0)
            stats[k]['f1']  += f1  * imgs.size(0)
            stats[k]['cm']  += cm
        total += imgs.size(0)
    for k in stats.keys():
        stats[k]['acc'] /= max(1, total)
        stats[k]['f1']  /= max(1, total)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k in stats.keys():
            np.save(os.path.join(out_dir, f"cm_{k}.npy"), stats[k]['cm'])
            try:
                labs = [str(i) for i in range(stats[k]['cm'].shape[0])]
                plot_cm_png(stats[k]['cm'], labs, os.path.join(out_dir, f"cm_{k}.png"))
            except Exception as e:
                print(f"[warn] plot cm failed for {k}: {e}")
    mean_acc = np.mean([stats[k]['acc'] for k in stats.keys()])
    return stats, float(mean_acc)


def main():
    ap = argparse.ArgumentParser()
    default_cfg = os.path.join(os.path.dirname(ROOT), 'µSHM-YOLO', 'yolov13_transformer_unified_v2_1.yaml')
    ap.add_argument("--cfg", required=False, type=str, default=default_cfg)
    ap.add_argument("--roi_root", required=True, type=str, help="包含 rois.json 或其子目录的根目录")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roi_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="runs/stage2_teacher")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighted_sample", type=int, default=1)
    ap.add_argument("--no_plot", type=int, default=1, help="禁用混淆矩阵 PNG 生成以避免 Windows 上的后端崩溃")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--label_smooth", type=float, default=0.1)
    ap.add_argument("--safe_loop", type=int, default=1, help="在 Windows/CPU 上启用无 DataLoader 的安全训练循环")
    args = ap.parse_args()
    # ==== Stabilize threading on Windows/CPU ====
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    set_seed(args.seed)
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
    n_total = len(ds); n_val = max(1, int(n_total * 0.2)); n_train = n_total - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    nw = max(0, int(args.num_workers))
    pin = (torch.device(args.device).type == 'cuda')
    if args.safe_loop:
        dl_train = None
        dl_val   = None
    else:
        if args.weighted_sample:
            wts = compute_species_weights(ds_train)
            sampler = WeightedRandomSampler(weights=wts, num_samples=len(wts), replacement=True)
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=nw, pin_memory=pin)
        else:
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    device = torch.device(args.device)
    model = MultiHeadClassifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = LabelSmoothingCE(eps=args.label_smooth).to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_mean_acc,val_species_acc,val_cell_org_acc,val_shape_acc,val_flagella_acc,val_chloro_acc\n")

    start_epoch = 1; best_mean_acc = -1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optim'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[resume] from {args.resume}, next epoch {start_epoch}")

    early = EarlyStopping(patience=args.patience, min_delta=1e-4)

    def _train_epoch_safe(ds_subset):
        model.train(); ls_meter=[]
        n = len(ds_subset); bs = max(1, args.batch_size)
        for start in range(0, n, bs):
            end = min(n, start + bs)
            batch_imgs=[]; sp=[]; co=[]; sh=[]; fl=[]; ch=[]
            for j in range(start, end):
                img_t, labels, _ = ds_subset[j]
                batch_imgs.append(img_t)
                sp.append(labels['species']); co.append(labels['cell_org']); sh.append(labels['shape']); fl.append(labels['flagella']); ch.append(labels['chloroplast'])
            imgs = torch.stack(batch_imgs, dim=0).to(device)
            sp = torch.stack(sp).to(device); co=torch.stack(co).to(device); sh=torch.stack(sh).to(device); fl=torch.stack(fl).to(device); ch=torch.stack(ch).to(device)
            outs = model(imgs)
            loss = (ce(outs['species'], sp) + ce(outs['cell_org'], co) + ce(outs['shape'], sh) + ce(outs['flagella'], fl) + ce(outs['chloroplast'], ch))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); ls_meter.append(loss.item())
        return float(np.mean(ls_meter)) if ls_meter else 0.0

    def _eval_safe(ds_subset, out_dir=None):
        model.eval(); heads={'species':45,'cell_org':4,'shape':10,'flagella':5,'chloroplast':4}
        stats = {k:{'acc':0,'f1':0,'cm':np.zeros((C,C),dtype=np.int64)} for k,C in heads.items()}
        total=0
        with torch.no_grad():
            n=len(ds_subset); bs=max(1, args.batch_size)
            for start in range(0, n, bs):
                end = min(n, start+bs)
                batch_imgs=[]; sp=[]; co=[]; sh=[]; fl=[]; ch=[]
                for j in range(start, end):
                    img_t, labels, _ = ds_subset[j]
                    batch_imgs.append(img_t)
                    sp.append(labels['species']); co.append(labels['cell_org']); sh.append(labels['shape']); fl.append(labels['flagella']); ch.append(labels['chloroplast'])
                imgs=torch.stack(batch_imgs,0).to(device)
                sp=torch.stack(sp).to(device); co=torch.stack(co).to(device); sh=torch.stack(sh).to(device); fl=torch.stack(fl).to(device); ch=torch.stack(ch).to(device)
                outs=model(imgs)
                for k, targ, C in [('species',sp,45),('cell_org',co,4),('shape',sh,10),('flagella',fl,5),('chloroplast',ch,4)]:
                    acc=accuracy(outs[k], targ); f1=macro_f1(outs[k], targ, C); cm=confusion_matrix(outs[k], targ, C)
                    stats[k]['acc']+=acc*imgs.size(0); stats[k]['f1']+=f1*imgs.size(0); stats[k]['cm']+=cm
                total+=imgs.size(0)
        for k in stats.keys():
            stats[k]['acc']/=max(1,total); stats[k]['f1']/=max(1,total)
        if out_dir and not args.no_plot:
            os.makedirs(out_dir, exist_ok=True)
            for k in stats.keys():
                np.save(os.path.join(out_dir, f"cm_{k}.npy"), stats[k]['cm'])
                try:
                    labs=[str(i) for i in range(stats[k]['cm'].shape[0])]
                    plot_cm_png(stats[k]['cm'], labs, os.path.join(out_dir, f"cm_{k}.png"))
                except Exception as e:
                    print(f"[warn] plot cm failed for {k}: {e}")
        mean_acc = np.mean([stats[k]['acc'] for k in stats.keys()])
        return stats, float(mean_acc)

    for epoch in range(start_epoch, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if args.safe_loop:
            tr_loss = _train_epoch_safe(ds_train)
            stats, mean_acc = _eval_safe(ds_val, out_dir=None if args.no_plot else os.path.join(args.out_dir, "cms"))
        else:
            tr_loss = train_one_epoch(model, dl_train, optimizer, device, ce, log_every=50)
            stats, mean_acc = evaluate(model, dl_val, device, out_dir=None if args.no_plot else os.path.join(args.out_dir, "cms"))
        line = f"{epoch},{tr_loss:.4f},{mean_acc:.4f},{stats['species']['acc']:.4f},{stats['cell_org']['acc']:.4f},{stats['shape']['acc']:.4f},{stats['flagella']['acc']:.4f},{stats['chloroplast']['acc']:.4f}"
        print("  " + line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
                   os.path.join(args.out_dir, "last_teacher.pt"))
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_teacher.pt"))
            print(f"  [best] mean_acc={best_mean_acc:.4f} -> saved best_teacher.pt")

        early.step(mean_acc)
        if early.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()