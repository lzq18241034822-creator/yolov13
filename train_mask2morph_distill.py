# -*- coding: utf-8 -*-
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

# 让脚本可从仓库根导入 datasets/ 与 models/ 与 utils/
ROOT = os.path.dirname(os.path.dirname(__file__))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 环境稳定性（Windows/CPU）
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
torch.set_num_threads(1)

try:
    from datasets.roi_with_teacher_logits import ROIWithTeacherDataset
except ModuleNotFoundError:
    import importlib.util as _ilu
    _ds_path = os.path.join(ROOT, 'datasets', 'roi_with_teacher_logits.py')
    _spec = _ilu.spec_from_file_location('datasets.roi_with_teacher_logits', _ds_path)
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    ROIWithTeacherDataset = _mod.ROIWithTeacherDataset

try:
    from models.multitask_heads import MultiHeadClassifier
except ModuleNotFoundError:
    import importlib.util as _ilu2
    _m_path = os.path.join(ROOT, 'models', 'multitask_heads.py')
    _m_spec = _ilu2.spec_from_file_location('models.multitask_heads', _m_path)
    _m_mod = _ilu2.module_from_spec(_m_spec); _m_spec.loader.exec_module(_m_mod)
    MultiHeadClassifier = _m_mod.MultiHeadClassifier

try:
    from utils.metrics_cls import accuracy, macro_f1, confusion_matrix
except ModuleNotFoundError:
    import importlib.util as _ilu3
    _u_path = os.path.join(ROOT, 'utils', 'metrics_cls.py')
    _u_spec = _ilu3.spec_from_file_location('utils.metrics_cls', _u_path)
    _u_mod = _ilu3.module_from_spec(_u_spec); _u_spec.loader.exec_module(_u_mod)
    accuracy = _u_mod.accuracy; macro_f1 = _u_mod.macro_f1; confusion_matrix = _u_mod.confusion_matrix

try:
    from utils.distill import kl_div_with_temperature
except ModuleNotFoundError:
    import importlib.util as _ilu4
    _d_path = os.path.join(ROOT, 'utils', 'distill.py')
    _d_spec = _ilu4.spec_from_file_location('utils.distill', _d_path)
    _d_mod = _ilu4.module_from_spec(_d_spec); _d_spec.loader.exec_module(_d_mod)
    kl_div_with_temperature = _d_mod.kl_div_with_temperature


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta
        self.best=None; self.count=0; self.should_stop=False
    def step(self, metric):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric; self.count=0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop=True


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__(); self.eps=eps; self.lsm=nn.LogSoftmax(dim=1)
    def forward(self, logits, target):
        C = logits.size(1)
        logp = self.lsm(logits)
        with torch.no_grad():
            true = torch.zeros_like(logp)
            true.fill_(self.eps/(C-1))
            true.scatter_(1, target.unsqueeze(1), 1-self.eps)
        return torch.mean(torch.sum(-true * logp, dim=1))


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_species_weights(ds):
    from collections import Counter
    cnt = Counter()
    for i in range(len(ds)):
        _, labels, _, _ = ds[i]
        cnt[int(labels['species'])] += 1
    total = sum(cnt.values())
    cw = {k: total / (v + 1e-6) for k, v in cnt.items()}
    weights = []
    for i in range(len(ds)):
        _, labels, _, _ = ds[i]
        weights.append(cw[int(labels['species'])])
    return torch.DoubleTensor(weights)


@torch.no_grad()
def evaluate(model, loader, device, out_dir=None):
    model.eval()
    heads = {'species':45, 'cell_org':4, 'shape':10, 'flagella':5, 'chloroplast':4}
    stats = {k: {'acc':0,'f1':0,'cm':np.zeros((C,C),dtype=np.int64)} for k,C in heads.items()}
    total=0
    for imgs, labels, _, meta in loader:
        imgs = imgs.to(device)
        sp = labels['species'].to(device)
        co = labels['cell_org'].to(device)
        sh = labels['shape'].to(device)
        fl = labels['flagella'].to(device)
        ch = labels['chloroplast'].to(device)
        outs = model(imgs)
        for k, targ, C in [('species',sp,45),('cell_org',co,4),('shape',sh,10),('flagella',fl,5),('chloroplast',ch,4)]:
            acc = accuracy(outs[k], targ); f1 = macro_f1(outs[k], targ, C); cm = confusion_matrix(outs[k], targ, C)
            stats[k]['acc'] += acc * imgs.size(0); stats[k]['f1'] += f1 * imgs.size(0); stats[k]['cm'] += cm
        total += imgs.size(0)
    for k in stats.keys():
        stats[k]['acc'] /= max(1,total); stats[k]['f1'] /= max(1,total)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k in stats.keys():
            np.save(os.path.join(out_dir, f"cm_{k}.npy"), stats[k]['cm'])
    mean_acc = np.mean([stats[k]['acc'] for k in stats.keys()])
    return stats, float(mean_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--roi_root", required=True)
    ap.add_argument("--teacher_npz", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roi_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="runs/stage2_student")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weighted_sample", type=int, default=1)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--label_smooth", type=float, default=0.1)
    ap.add_argument("--kd_weight", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--safe_loop", type=int, default=1, help="在 Windows/CPU 上启用无 DataLoader 的安全训练循环")
    ap.add_argument("--no_plot", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    ds = ROIWithTeacherDataset(args.roi_root, args.cfg, args.teacher_npz, roi_size=args.roi_size)

    # 固定划分 8:2
    n_total = len(ds); n_val = max(1, int(0.2*n_total)); n_train = max(1, n_total - n_val)

    pin = (args.device.lower().startswith('cuda') and torch.cuda.is_available())
    if args.safe_loop:
        dl_train = None; dl_val = None
        idxs = np.arange(len(ds)); np.random.seed(args.seed); np.random.shuffle(idxs)
        val_idx = idxs[:n_val]; tr_idx = idxs[n_val:]
        def batch_iter(idxs, bs):
            for i in range(0, len(idxs), bs):
                yield [ds[j] for j in idxs[i:i+bs]]
    else:
        ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
        if args.weighted_sample:
            wts = compute_species_weights(ds_train.dataset if hasattr(ds_train,'dataset') else ds_train)
            sampler = WeightedRandomSampler(weights=wts, num_samples=len(wts), replacement=True)
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
        else:
            dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

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
        print(f"[resume] -> epoch {start_epoch}")
    early = EarlyStopping(patience=args.patience, min_delta=1e-4)

    for epoch in range(start_epoch, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train(); losses=[]
        if args.safe_loop:
            for batch in batch_iter(tr_idx, args.batch_size):
                imgs = torch.stack([b[0] for b in batch]).to(device)
                labels = {k: torch.stack([b[1][k] for b in batch]).to(device) for k in ['species','cell_org','shape','flagella','chloroplast']}
                tlogits = {k: torch.stack([b[2][k] for b in batch]).to(device) for k in ['species','cell_org','shape','flagella','chloroplast']}
                outs = model(imgs)
                loss_ce = (ce(outs['species'], labels['species']) +
                           ce(outs['cell_org'], labels['cell_org']) +
                           ce(outs['shape'], labels['shape']) +
                           ce(outs['flagella'], labels['flagella']) +
                           ce(outs['chloroplast'], labels['chloroplast']))
                loss_kd = (kl_div_with_temperature(outs['species'], tlogits['species'], args.kd_T) +
                           kl_div_with_temperature(outs['cell_org'], tlogits['cell_org'], args.kd_T) +
                           kl_div_with_temperature(outs['shape'], tlogits['shape'], args.kd_T) +
                           kl_div_with_temperature(outs['flagella'], tlogits['flagella'], args.kd_T) +
                           kl_div_with_temperature(outs['chloroplast'], tlogits['chloroplast'], args.kd_T))
                loss = loss_ce + args.kd_weight * loss_kd
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(loss.item())
        else:
            for imgs, labels, tlogits, meta in dl_train:
                imgs = imgs.to(device)
                for k in labels: labels[k]=labels[k].to(device)
                for k in tlogits: tlogits[k]=tlogits[k].to(device)
                outs = model(imgs)
                loss_ce = (ce(outs['species'], labels['species']) +
                           ce(outs['cell_org'], labels['cell_org']) +
                           ce(outs['shape'], labels['shape']) +
                           ce(outs['flagella'], labels['flagella']) +
                           ce(outs['chloroplast'], labels['chloroplast']))
                loss_kd = (kl_div_with_temperature(outs['species'], tlogits['species'], args.kd_T) +
                           kl_div_with_temperature(outs['cell_org'], tlogits['cell_org'], args.kd_T) +
                           kl_div_with_temperature(outs['shape'], tlogits['shape'], args.kd_T) +
                           kl_div_with_temperature(outs['flagella'], tlogits['flagella'], args.kd_T) +
                           kl_div_with_temperature(outs['chloroplast'], tlogits['chloroplast'], args.kd_T))
                loss = loss_ce + args.kd_weight * loss_kd
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                losses.append(loss.item())

        tr_loss = float(np.mean(losses)) if losses else 0.0

        # 验证
        if args.safe_loop:
            model.eval()
            accs = {'species':0,'cell_org':0,'shape':0,'flagella':0,'chloroplast':0}
            total=0
            with torch.no_grad():
                for batch in batch_iter(val_idx, args.batch_size):
                    imgs = torch.stack([b[0] for b in batch]).to(device)
                    labels = {k: torch.stack([b[1][k] for b in batch]).to(device) for k in accs.keys()}
                    outs = model(imgs)
                    accs['species'] += accuracy(outs['species'], labels['species'])*imgs.size(0)
                    accs['cell_org'] += accuracy(outs['cell_org'], labels['cell_org'])*imgs.size(0)
                    accs['shape'] += accuracy(outs['shape'], labels['shape'])*imgs.size(0)
                    accs['flagella'] += accuracy(outs['flagella'], labels['flagella'])*imgs.size(0)
                    accs['chloroplast'] += accuracy(outs['chloroplast'], labels['chloroplast'])*imgs.size(0)
                    total += imgs.size(0)
            for k in accs: accs[k] /= max(1,total)
            mean_acc = float(np.mean(list(accs.values())))
            stats = {k:{'acc':accs[k]} for k in accs}
        else:
            stats, mean_acc = evaluate(model, dl_val, device, out_dir=None if args.no_plot else os.path.join(args.out_dir, "cms"))

        # 记录与保存
        print(f"  loss={tr_loss:.4f} mean_acc={mean_acc:.4f} sp={stats['species']['acc']:.4f} co={stats['cell_org']['acc']:.4f}")
        with open(os.path.join(args.out_dir, "log.csv"), 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{tr_loss:.4f},{mean_acc:.4f},{stats['species']['acc']:.4f},{stats['cell_org']['acc']:.4f},"
                    f"{stats.get('shape',{}).get('acc',0):.4f},{stats.get('flagella',{}).get('acc',0):.4f},{stats.get('chloroplast',{}).get('acc',0):.4f}\n")

        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
                   os.path.join(args.out_dir, "last_student.pt"))
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_student.pt"))
            print(f"  [best] mean_acc={best_mean_acc:.4f} -> saved best_student.pt")

        early.step(mean_acc)
        if early.should_stop:
            print(f"  Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()