# -*- coding: utf-8 -*-
import os, glob, random, argparse
random.seed(42)

def main(root, val_ratio=0.2, out_dir="splits"):
    img_dir = os.path.join(root, "images", "train")
    stems = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"):
        for p in glob.glob(os.path.join(img_dir, ext)):
            stems.append(os.path.splitext(os.path.basename(p))[0])
    stems = sorted(list(set(stems)))
    n = len(stems)
    n_val = max(1, int(n * val_ratio))
    random.shuffle(stems)
    val_ids = set(stems[:n_val])
    tr_ids = [s for s in stems if s not in val_ids]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tr_ids))
    with open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(val_ids)))
    print(f"Saved {len(tr_ids)} train and {len(val_ids)} val to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--out_dir", type=str, default="splits")
    args = ap.parse_args()
    main(args.data_root, args.val_ratio, args.out_dir)