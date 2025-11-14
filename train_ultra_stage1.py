# -*- coding: utf-8 -*-
"""
Ultralytics SegmentTrainer 快速对接脚本（Stage1 基线）

功能概述：
- 读取 µSHM-YOLO 统一标签（每行 5ID + polygon），转换为 Ultralytics 支持的 segmentation 标签（单类：0 + polygon）
- 运行时猴补 ultralytics.data.utils.img2label_paths，将 images/* 关联到 labels-ultra/*
- 使用本地 yolov13-main/ultralytics 的 SegmentationTrainer（经 YOLO API）训练并评估，输出 mAP/APS 等指标

使用示例：
    python tools/train_ultra_stage1.py --epochs 1 --imgsz 256 --batch 2

说明：
- 不修改原有 samples/labels 下的统一标签，转换输出至 samples/labels-ultra 以共存
- 需存在本仓库同级的 yolov13-main 目录（包含 ultralytics 模块）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ULTRALYTICS_IGNORE_WALL_ERRORS", "1")
import sys
import glob
import shutil
import importlib
import traceback
from typing import List
import numpy as np

def repo_paths():
    cur = os.path.dirname(__file__)
    repo_root = os.path.dirname(cur)  # µSHM-YOLO 根
    top_root = os.path.dirname(repo_root)  # G:\yoloV13
    ultra_root = os.path.join(top_root, 'yolov13-main')
    return repo_root, top_root, ultra_root

REPO_ROOT, TOP_ROOT, ULTRA_ROOT = repo_paths()

if ULTRA_ROOT not in sys.path:
    sys.path.append(ULTRA_ROOT)

from ruamel.yaml import YAML
import cv2


def parse_line_unified(s: str):
    parts = s.strip().split()
    if len(parts) < 11:
        return None, None
    try:
        ids5 = list(map(int, parts[:5]))
        seg = list(map(float, parts[5:]))
        if len(seg) % 2 != 0:
            return None, None
        return ids5, seg
    except Exception:
        return None, None


def convert_labels_to_ultra(src_dir: str, dst_dir: str, skip_species: List[int] = None):
    os.makedirs(dst_dir, exist_ok=True)
    for txt in glob.glob(os.path.join(src_dir, '*.txt')):
        stem = os.path.splitext(os.path.basename(txt))[0]
        outp = os.path.join(dst_dir, stem + '.txt')
        lines_out = []
        with open(txt, 'r', encoding='utf-8') as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith('#'):
                    continue
                ids5, seg = parse_line_unified(s)
                if ids5 is None:
                    continue
                species = int(ids5[0])
                if skip_species and species in skip_species:
                    continue
                # 单类分割：cls=0 + polygon（归一化坐标）
                lines_out.append('0 ' + ' '.join(f'{v:.6f}' for v in seg))
        # 若为空则仍写空文件以通过校验
        with open(outp, 'w', encoding='utf-8') as fw:
            fw.write('\n'.join(lines_out))


def build_labels_ultra(samples_root: str, cfg_yaml_path: str):
    """
    从 samples/labels/* 生成 samples/labels-ultra/*
    跳过 YAML 中配置的 skip_species_ids（默认 [40,41,42,43]）
    """
    yaml = YAML()
    try:
        with open(cfg_yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f)
        skip_species = cfg['training']['stage1_training']['dataloader']['stage1_class_filter'].get(
            'skip_species_ids', [40, 41, 42, 43]
        )
    except Exception:
        skip_species = [40, 41, 42, 43]

    for split in ('train', 'val', 'test'):
        src = os.path.join(samples_root, 'labels', split)
        dst = os.path.join(samples_root, 'labels-ultra', split)
        os.makedirs(dst, exist_ok=True)
        convert_labels_to_ultra(src, dst, skip_species=skip_species)


def ensure_data_yaml(samples_root: str) -> str:
    """生成或覆盖 samples/data-ultra-seg.yaml（nc=1 单类）"""
    data_yaml = os.path.join(samples_root, 'data-ultra-seg.yaml')
    data = {
        'path': samples_root,
        'train': os.path.join(samples_root, 'images', 'train'),
        'val': os.path.join(samples_root, 'images', 'val'),
        'test': os.path.join(samples_root, 'images', 'test'),
        'nc': 1,
        'names': {0: 'foreground'}
    }
    yaml = YAML()
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
    return data_yaml


def ensure_data_yaml_from_lists(samples_root: str) -> str:
    """根据 images/*.txt 列表生成 samples/data-ultra-seg-lists.yaml。"""
    data_yaml = os.path.join(samples_root, 'data-ultra-seg-lists.yaml')
    images_dir = os.path.join(samples_root, 'images')
    train_list = os.path.join(images_dir, 'train.txt')
    val_list = os.path.join(images_dir, 'val.txt')
    test_list = os.path.join(images_dir, 'test.txt')
    data = {
        'path': samples_root,
        'train': train_list,
        'val': val_list,
        'test': test_list,
        'nc': 1,
        'names': {0: 'foreground'}
    }
    yaml = YAML()
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)
    return data_yaml


def convert_jpegs_to_png(samples_root: str):
    """将 samples/images/* 中的 .jpg/.jpeg 转换为 .png 并删除原文件，避免 JPEG 修复触发崩溃。"""
    for split in ('train', 'val', 'test'):
        img_dir = os.path.join(samples_root, 'images', split)
        if not os.path.isdir(img_dir):
            continue
        for ext in ('.jpg', '.jpeg', '.JPG', '.JPEG'):
            for jp in glob.glob(os.path.join(img_dir, f'*{ext}')):
                stem = os.path.splitext(os.path.basename(jp))[0]
                outp = os.path.join(img_dir, stem + '.png')
                try:
                    im = cv2.imread(jp, cv2.IMREAD_UNCHANGED)
                    if im is None:
                        continue
                    cv2.imwrite(outp, im)
                    os.remove(jp)
                except Exception:
                    # 若某些图像无法读取，跳过即可，Ultralytics 会在扫描阶段报告
                    pass


def remove_ultra_caches(samples_root: str):
    """删除 labels-ultra/* 下的 *.cache 文件，强制使用最新猴补重建缓存。"""
    for split in ('train', 'val', 'test'):
        cache_path = os.path.join(samples_root, 'labels-ultra', f'{split}.cache')
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
        except Exception:
            pass

def build_image_list_txt(samples_root: str):
    """为 train/val/test 生成仅包含可读取图像的 .txt 列表，优先使用 PNG。"""
    from PIL import Image
    images_dir = os.path.join(samples_root, 'images')
    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(images_dir, split)
        if not os.path.isdir(split_dir):
            # 若不存在该 split 目录则跳过
            continue
        out_txt = os.path.join(images_dir, f'{split}.txt')
        ok_paths = []
        # 1) 优先收集 PNG
        for root, _, files in os.walk(split_dir):
            for fn in files:
                if fn.lower().endswith('.png'):
                    p = os.path.join(root, fn)
                    try:
                        with Image.open(p) as im:
                            im.verify()
                        rel = os.path.relpath(p, images_dir).replace('\\', '/')
                        ok_paths.append('./' + rel)
                    except Exception:
                        pass
        # 2) 如无 PNG，则尝试筛选可正常打开的 JPG/JPEG
        if not ok_paths:
            for root, _, files in os.walk(split_dir):
                for fn in files:
                    if fn.lower().endswith(('.jpg', '.jpeg')):
                        p = os.path.join(root, fn)
                        try:
                            with Image.open(p) as im:
                                im.verify()
                            rel = os.path.relpath(p, images_dir).replace('\\', '/')
                            ok_paths.append('./' + rel)
                        except Exception:
                            pass
        # 始终写入（即便为空），避免加载目录时再次扫描 JPEG
        with open(out_txt, 'w', encoding='utf-8') as f:
            for p in sorted(ok_paths):
                f.write(p + '\n')


def monkey_patch_img2label_to_ultra():
    """将 images/* → labels-ultra/* 的路径映射猴补到 Ultralytics 运行环境。"""
    from ultralytics.data import utils as uutils

    def img2label_paths_ultra(img_paths):
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels-ultra{os.sep}"
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    # 覆盖模块函数本体
    uutils.img2label_paths = img2label_paths_ultra
    # 同时更新 dataset 模块内的引用
    udata = importlib.import_module('ultralytics.data.dataset')
    setattr(udata, 'img2label_paths', img2label_paths_ultra)


def monkey_patch_disable_jpeg_restore():
    """禁用 Ultralytics 在 verify_image_label 中对损坏 JPEG 的修复保存，避免 Windows/Pillow 崩溃路径。"""
    from ultralytics.data import utils as uutils
    from ultralytics.utils.ops import segments2boxes
    import os as _os
    from PIL import Image as _Image

    def _verify_image_label_nojpeg(args):
        im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
        nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
        try:
            # Verify images (no JPEG repair)
            im = _Image.open(im_file)
            im.verify()
            shape = uutils.exif_size(im)
            shape = (shape[1], shape[0])
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in uutils.IMG_FORMATS, f"invalid image format {im.format}. {uutils.FORMATS_HELP_MSG}"

            # Verify labels
            if _os.path.isfile(lb_file):
                nf = 1
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            else:
                lb = []

            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                max_cls = lb[:, 0].max()
                assert max_cls <= num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)

            if keypoint:
                keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
                if ndim == 2:
                    kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)
            lb = lb[:, :5]
            return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
            return [None, None, None, None, None, nm, nf, ne, nc, msg]

    # 覆盖 utils 和 dataset 中的同名引用
    uutils.verify_image_label = _verify_image_label_nojpeg
    udata = importlib.import_module('ultralytics.data.dataset')
    setattr(udata, 'verify_image_label', _verify_image_label_nojpeg)


def monkey_patch_load_image_with_pil():
    """将 BaseDataset.load_image 替换为基于 PIL 的读取，避免 OpenCV 在 Windows 上的本地崩溃。"""
    import importlib as _importlib
    from PIL import Image as _Image
    import numpy as _np
    _base = _importlib.import_module('ultralytics.data.base')

    def _load_image_pil(self, i, rect_mode=True):
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:
            if fn.exists():
                try:
                    im = _np.load(fn)
                except Exception:
                    # 回退到 PIL 读取
                    with _Image.open(f) as pil:
                        pil = pil.convert('RGB')
                        im = _np.array(pil)[:, :, ::-1]  # RGB->BGR
            else:
                with _Image.open(f) as pil:
                    pil = pil.convert('RGB')
                    im = _np.array(pil)[:, :, ::-1]  # RGB->BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(int(_np.ceil(w0 * r)), self.imgsz), min(int(_np.ceil(h0 * r)), self.imgsz))
                    # 使用 PIL resize 再转回 numpy，可避免 OpenCV 调用
                    pil = _Image.fromarray(im[:, :, ::-1])  # BGR->RGB
                    pil = pil.resize((w, h), resample=_Image.BILINEAR)
                    im = _np.array(pil)[:, :, ::-1]
            elif not (h0 == w0 == self.imgsz):
                pil = _Image.fromarray(im[:, :, ::-1])
                pil = pil.resize((self.imgsz, self.imgsz), resample=_Image.BILINEAR)
                im = _np.array(pil)[:, :, ::-1]

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != 'ram':
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    setattr(_base.BaseDataset, 'load_image', _load_image_pil)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--model', type=str, default=os.path.join(ULTRA_ROOT, 'ultralytics', 'cfg', 'models', '11', 'yolo11n-seg.yaml'))
    ap.add_argument('--cfg', type=str, default=os.path.join(REPO_ROOT, 'yolov13_transformer_unified_v2_1.yaml'))
    args = ap.parse_args()

    samples_root = os.path.join(REPO_ROOT, 'samples')
    reports_dir = os.path.join(REPO_ROOT, 'tools', 'reports_ultra')
    os.makedirs(reports_dir, exist_ok=True)

    # 1) 生成 labels-ultra/*
    build_labels_ultra(samples_root, cfg_yaml_path=args.cfg)

    # 2) 数据 YAML（nc=1）
    data_yaml = ensure_data_yaml(samples_root)

    # 3) 猴补路径映射
    monkey_patch_img2label_to_ultra()
    # 3.1) 禁用 JPEG 修复保存（避免 Windows/Pillow 崩溃路径）
    monkey_patch_disable_jpeg_restore()

    # 3.2) 主动将 JPEG 转换为 PNG，彻底规避 JPEG 修复路径
    convert_jpegs_to_png(samples_root)
    # 3.3) 清理旧缓存，确保新验证逻辑生效
    remove_ultra_caches(samples_root)
    # 3.4) 生成 train/val/test 的图像列表 .txt，仅包含可读取图像
    build_image_list_txt(samples_root)
    # 3.5) 将数据加载改为 PIL，规避 OpenCV 读取触发的 Windows 本地崩溃
    monkey_patch_load_image_with_pil()

    # 4) 训练与评估
    from ultralytics import YOLO
    model = YOLO(args.model)
    # 若存在图像列表，则优先使用 .txt 方案（写入新的 data yaml）
    images_dir = os.path.join(samples_root, 'images')
    train_list = os.path.join(images_dir, 'train.txt')
    val_list = os.path.join(images_dir, 'val.txt')
    test_list = os.path.join(images_dir, 'test.txt')
    data_for_train = data_yaml
    if os.path.exists(train_list):
        data_for_train = ensure_data_yaml_from_lists(samples_root)
    try:
        model.train(
            data=data_for_train,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=False,
            rect=True,
            augment=False,
            amp=False,
            val=False,
            pretrained=False,
            project=reports_dir,
            name='stage1_ultra',
        )
    except SystemExit as e:
        print(f"[ERROR] 训练阶段 SystemExit: code={getattr(e, 'code', None)}")
        try:
            traceback.print_exc()
        except Exception:
            pass
    except Exception as e:
        print("[ERROR] 训练阶段抛出异常:", e)
        traceback.print_exc()

    # 可选：仅当 val.txt 非空时触发一次 val 输出指标概要
    # 仅当使用列表 YAML 且 val.txt 非空时触发一次 val 输出指标概要
    try:
        lists_yaml_used = os.path.exists(os.path.join(samples_root, 'data-ultra-seg-lists.yaml'))
        do_val = lists_yaml_used and os.path.getsize(os.path.join(samples_root, 'images', 'val.txt')) > 0
    except Exception:
        do_val = False
    if do_val:
        try:
            model.val(
                data=ensure_data_yaml_from_lists(samples_root),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                project=reports_dir,
                name='stage1_ultra_val'
            )
        except SystemExit as e:
            print(f"[ERROR] 验证阶段 SystemExit: code={getattr(e, 'code', None)}")
            try:
                traceback.print_exc()
            except Exception:
                pass
        except Exception as e:
            print("[ERROR] 验证阶段抛出异常:", e)
            traceback.print_exc()

    print(f"[OK] 训练完成。日志与权重位于：{reports_dir}")


if __name__ == '__main__':
    main()