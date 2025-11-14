# -*- coding: utf-8 -*-
# 统一标注 GUI：Polygon + Rect + 5ID（按 YAML 自动注入头部与命名）
# 运行:
#   python tools/unified_annotator_gui.py \
#       --image_dir images/train \
#       --output_dir labels/train \
#       --cfg g:\yoloV13\µSHM-YOLO\yolov13_transformer_unified_v2_1.yaml

import os
import sys
import glob
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ruamel.yaml import YAML

from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QListWidget, QListWidgetItem,
    QMessageBox, QSplitter
)


@dataclass
class Annotation:
    kind: str  # 'polygon' or 'rect'
    points: List[Tuple[float, float]] = field(default_factory=list)  # in image px coords
    rect: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h in image px
    species: int = 0
    cell_org: int = 7
    shape: int = 12
    flagella: int = 22
    chloroplast: int = 27


class Config:
    def __init__(self, cfg_path: str):
        yaml = YAML()
        with open(cfg_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f)
        self.naming = self.cfg['dataset']['naming']
        self.number_format = self.naming.get('number_format', {})
        self.microscope = self.cfg.get('microscope', {})
        self.classes = self.cfg['classes']
        self.groups = self.classes.get('groups', {})
        self.names = self.classes.get('names', {})
        self.names_zh = self.classes.get('names_zh', {})

    def id_name(self, cid: int) -> str:
        return str(self.names_zh.get(cid, self.names.get(cid, cid)))

    def build_label_filename(self, image_stem: str) -> str:
        # 按需求：标签文件名与图片同名，仅扩展名改为 .txt
        return f"{image_stem}.txt"

    def header_lines(self, img_w: int, img_h: int, img_stats: Optional[dict] = None) -> List[str]:
        hdr = self.naming.get('header_lines', [])
        nf = self.number_format.get('pixel_size_um', {'digits': 4, 'trim_trailing_zeros': False})
        digits = nf.get('digits', 4)
        trim = nf.get('trim_trailing_zeros', False)
        px = f"{float(self.microscope.get('pixel_size_um', 0.0)):.{digits}f}"
        if trim and '.' in px:
            px = px.rstrip('0').rstrip('.')
        # 基本字段
        mapping = {
            '{magnification}': self.microscope.get('magnification', ''),
            '{pixel_size_um}': px,
            '{pixel_dimensions_px}': f"{img_w}x{img_h}",
            '{magnification_camera}': self.microscope.get('magnification_camera', ''),
            '{pixel_size_um_source}': self.microscope.get('pixel_size_um_source', ''),
        }
        # 像素统计（可选注入）
        if img_stats:
            mapping.update({
                '{pixel_mean_rgb}': img_stats.get('pixel_mean_rgb', ''),
                '{pixel_std_rgb}': img_stats.get('pixel_std_rgb', ''),
                '{gray_percentiles_1_99}': img_stats.get('gray_percentiles_1_99', ''),
                '{dtype}': img_stats.get('dtype', ''),
                '{dynamic_range_01}': img_stats.get('dynamic_range_01', ''),
            })
        outs = []
        for line in hdr:
            s = line
            for k, v in mapping.items():
                s = s.replace(k, str(v))
            outs.append(s)
        return outs


class Canvas(QWidget):
    # 选择改变信号（在类级别定义，符合 PyQt 规范）
    selection_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.image = None  # BGR numpy
        self.qpix = None
        self.scale = 1.0
        self.annotations: List[Annotation] = []
        self.current_poly: List[Tuple[float, float]] = []
        self.mode = 'polygon'  # 'polygon' or 'rect'
        self.drawing_rect = False
        self.rect_start: Optional[Tuple[int, int]] = None
        self.rect_end: Optional[Tuple[int, int]] = None
        self.selected_index: Optional[int] = None
        self.default_classes = (0, 7, 12, 22, 27)
        self.setMouseTracking(True)

    def set_image(self, img: np.ndarray):
        self.image = img
        h, w = img.shape[:2]
        self.scale = min(1200 / max(w, 1), 800 / max(h, 1))
        disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        disp = cv2.resize(disp, (int(w * self.scale), int(h * self.scale)))
        h2, w2 = disp.shape[:2]
        qimg = QImage(disp.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self.qpix = QPixmap.fromImage(qimg)
        self.setMinimumSize(w2, h2)
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.current_poly.clear()
        self.drawing_rect = False
        self.rect_start = None
        self.rect_end = None
        self.update()

    def set_default_classes(self, cls_tuple):
        self.default_classes = cls_tuple

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.qpix is not None:
            painter.drawPixmap(0, 0, self.qpix)

        pen_a = QPen(QColor(0, 255, 0), 2)
        pen_sel = QPen(QColor(255, 0, 0), 3)
        painter.setPen(pen_a)

        # draw saved annotations
        for idx, ann in enumerate(self.annotations):
            pen = pen_sel if self.selected_index == idx else pen_a
            painter.setPen(pen)
            if ann.kind == 'rect' and ann.rect is not None:
                x, y, w, h = ann.rect
                painter.drawRect(int(x * self.scale), int(y * self.scale), int(w * self.scale), int(h * self.scale))
            else:
                pts = [(int(px * self.scale), int(py * self.scale)) for (px, py) in ann.points]
                for i in range(len(pts)):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % len(pts)]
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # current drawing
        painter.setPen(QPen(QColor(255, 165, 0), 2))
        if self.mode == 'polygon' and self.current_poly:
            pts = [(int(px * self.scale), int(py * self.scale)) for (px, py) in self.current_poly]
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
        elif self.mode == 'rect' and self.rect_start and self.rect_end:
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.LeftButton:
            if self.mode == 'polygon':
                # 若当前未在绘制新多边形，优先尝试命中已有标注以进行选择/编辑
                if not self.current_poly:
                    hx = event.pos().x() / self.scale
                    hy = event.pos().y() / self.scale
                    hit = self.hit_test(hx, hy)
                    if hit is not None:
                        self.set_selected(hit)
                        self.selection_changed.emit(hit)
                        return
                # 否则进入新增点流程
                x = event.pos().x() / self.scale
                y = event.pos().y() / self.scale
                # Clamp 到图像边界
                h, w = self.image.shape[:2]
                x = max(0.0, min(x, float(w)))
                y = max(0.0, min(y, float(h)))
                self.current_poly.append((x, y))
                self.update()
            else:
                self.drawing_rect = True
                self.rect_start = (event.pos().x(), event.pos().y())
                self.rect_end = self.rect_start
                self.update()

    def mouseMoveEvent(self, event):
        if self.mode == 'rect' and self.drawing_rect and self.rect_start is not None:
            self.rect_end = (event.pos().x(), event.pos().y())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == 'rect' and self.drawing_rect and event.button() == Qt.LeftButton:
            self.rect_end = (event.pos().x(), event.pos().y())
            # finalize rect
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            rx = min(x1, x2) / self.scale
            ry = min(y1, y2) / self.scale
            rw = abs(x2 - x1) / self.scale
            rh = abs(y2 - y1) / self.scale
            # Clamp 到图像边界，避免出界
            h, w = self.image.shape[:2]
            rx = max(0.0, min(rx, float(w)))
            ry = max(0.0, min(ry, float(h)))
            # 右下角也需在边界内
            rx2 = max(0.0, min(rx + rw, float(w)))
            ry2 = max(0.0, min(ry + rh, float(h)))
            rw = max(0.0, rx2 - rx)
            rh = max(0.0, ry2 - ry)
            sp, co, sh, fl, ch = self.default_classes
            self.annotations.append(Annotation(kind='rect', rect=(rx, ry, rw, rh), species=sp, cell_org=co, shape=sh, flagella=fl, chloroplast=ch))
            self.drawing_rect = False
            self.rect_start = None
            self.rect_end = None
            self.update()

    def undo_point(self):
        if self.mode == 'polygon' and self.current_poly:
            self.current_poly.pop()
            self.update()

    def finish_polygon(self):
        if self.mode == 'polygon' and len(self.current_poly) >= 3:
            sp, co, sh, fl, ch = self.default_classes
            self.annotations.append(Annotation(kind='polygon', points=list(self.current_poly), species=sp, cell_org=co, shape=sh, flagella=fl, chloroplast=ch))
            # 完成后自动选中新创建的标注，便于立刻调整分类
            self.selected_index = len(self.annotations) - 1
            self.current_poly.clear()
            self.update()

    def clear_all(self):
        self.annotations.clear()
        self.current_poly.clear()
        self.update()

    def set_selected(self, idx: Optional[int]):
        self.selected_index = idx
        self.update()

    def update_selected_classes(self, sp: int, co: int, sh: int, fl: int, ch: int):
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.annotations):
            ann = self.annotations[self.selected_index]
            ann.species = sp; ann.cell_org = co; ann.shape = sh; ann.flagella = fl; ann.chloroplast = ch
            self.update()

    def point_in_polygon(self, x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-8) + x1)
            if cond:
                inside = not inside
        return inside

    def hit_test(self, x: float, y: float) -> Optional[int]:
        # 命中测试：优先多边形点内；其次矩形 bbox 内
        for i, ann in enumerate(self.annotations):
            if ann.kind == 'polygon' and len(ann.points) >= 3:
                if self.point_in_polygon(x, y, ann.points):
                    return i
            elif ann.kind == 'rect' and ann.rect is not None:
                rx, ry, rw, rh = ann.rect
                if (rx <= x <= rx + rw) and (ry <= y <= ry + rh):
                    return i
        return None

    def delete_selected(self):
        if self.selected_index is not None and 0 <= self.selected_index < len(self.annotations):
            self.annotations.pop(self.selected_index)
            self.selected_index = None
            self.update()


def list_images(image_dir: str) -> List[str]:
    # 支持更广泛的常见图片格式（忽略大小写）
    supported_exts = {
        '.jpg', '.jpeg', '.jpe', '.jfif',
        '.png', '.bmp', '.dib',
        '.tif', '.tiff',
        '.webp', '.jp2',
        '.ppm', '.pgm', '.pbm', '.pnm',
        '.sr', '.ras',
        '.gif', '.ico'
        # 注：HEIC/AVIF/RAW 等需安装额外插件，GUI 将尝试 Pillow 回退
    }
    files: List[str] = []
    for root, _, fnames in os.walk(image_dir):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in supported_exts:
                files.append(os.path.join(root, fn))
    return sorted(files)


def rect_to_yolo(img_w, img_h, x, y, w, h):
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    return max(0.0, min(1.0, xc)), max(0.0, min(1.0, yc)), max(0.0, min(1.0, w / img_w)), max(0.0, min(1.0, h / img_h))


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = float(min(xs))
    top = float(min(ys))
    right = float(max(xs))
    bottom = float(max(ys))
    return left, top, right - left, bottom - top


def read_image_any(path: str) -> Optional[np.ndarray]:
    """尝试读取任何常见格式图片：优先用 OpenCV，失败时用 Pillow 回退。
    返回 BGR numpy 数组；若为灰度或含 Alpha，将统一为 BGR。
    """
    # 先用 OpenCV 常规路径读取（快）。若失败，尝试 Windows/Unicode 兼容的 imdecode
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None:
        try:
            import numpy as _np
            data = _np.fromfile(path, dtype=_np.uint8)
            if data.size > 0:
                img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
    if img is not None:
        # 统一通道格式
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    # Pillow 回退
    try:
        from PIL import Image
        pil = Image.open(path)
        pil = pil.convert('RGB')
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def compute_image_stats(img: np.ndarray) -> dict:
    """计算像素统计（0-1归一化）：RGB均值/方差、灰度1/99分位、数据类型与动态范围。"""
    stats = {}
    dtype = str(img.dtype)
    stats['dtype'] = dtype
    # 归一化到0..1
    if img.dtype == np.uint8:
        norm = img.astype(np.float32) / 255.0
        min_v, max_v = 0.0, 1.0
    elif img.dtype == np.uint16:
        norm = img.astype(np.float32) / 65535.0
        min_v, max_v = 0.0, 1.0
    else:
        m = float(img.min())
        M = float(img.max())
        min_v, max_v = (0.0, 1.0) if M > m else (m, M)
        norm = (img.astype(np.float32) - m) / (M - m + 1e-8)

    if norm.ndim == 2:
        # 灰度视为 RGB 同值
        mean = [float(norm.mean())] * 3
        std = [float(norm.std())] * 3
        gray = norm
    else:
        # BGR → RGB 仅用于统计显示
        rgb = cv2.cvtColor((norm * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = [float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())]
        std = [float(rgb[:, :, 0].std()), float(rgb[:, :, 1].std()), float(rgb[:, :, 2].std())]
        gray = cv2.cvtColor((norm * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    stats['pixel_mean_rgb'] = f"{mean[0]:.4f},{mean[1]:.4f},{mean[2]:.4f}"
    stats['pixel_std_rgb'] = f"{std[0]:.4f},{std[1]:.4f},{std[2]:.4f}"
    stats['gray_percentiles_1_99'] = f"{p1:.4f},{p99:.4f}"
    stats['dynamic_range_01'] = f"{min_v:.4f},{max_v:.4f}"
    return stats


def save_labels(cfg: Config, image_path: str, output_dir: str, anns: List[Annotation], img_w: int, img_h: int, img_stats: Optional[dict] = None):
    os.makedirs(output_dir, exist_ok=True)
    filename = cfg.build_label_filename(os.path.splitext(os.path.basename(image_path))[0])
    label_path = os.path.join(output_dir, filename)
    lines: List[str] = []
    for ann in anns:
        if ann.kind == 'rect' and ann.rect is not None:
            x, y, w, h = ann.rect
            # 将矩形转为四点多边形，导出为点序列（不再输出 bbox 四元组）
            seg = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            # Clamp 到 [0,1]
            xs = [f"{max(0.0, min(1.0, px / img_w)):.6f}" for (px, _) in seg]
            ys = [f"{max(0.0, min(1.0, py / img_h)):.6f}" for (_, py) in seg]
        else:
            # Clamp 到 [0,1]
            xs = [f"{max(0.0, min(1.0, px / img_w)):.6f}" for (px, _) in ann.points]
            ys = [f"{max(0.0, min(1.0, py / img_h)):.6f}" for (_, py) in ann.points]
        cls_part = f"{ann.species} {ann.cell_org} {ann.shape} {ann.flagella} {ann.chloroplast}"
        seg_pairs = " ".join([f"{xs[i]} {ys[i]}" for i in range(len(xs))])
        lines.append(f"{cls_part} {seg_pairs}")

    with open(label_path, 'w', encoding='utf-8') as f:
        if cfg.naming.get('auto_inject_header', False):
            for hl in cfg.header_lines(img_w, img_h, img_stats):
                f.write(hl + '\n')
        for ln in lines:
            f.write(ln + '\n')
    return label_path


def load_labels_if_exist(cfg: Config, image_path: str, output_dir: str, img_w: int, img_h: int) -> List[Annotation]:
    filename = cfg.build_label_filename(os.path.splitext(os.path.basename(image_path))[0])
    label_path = os.path.join(output_dir, filename)
    anns: List[Annotation] = []
    if not os.path.exists(label_path):
        return anns
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # 基本：五个分类整数
            if len(parts) < 5:
                continue
            try:
                sp = int(parts[0]); co = int(parts[1]); sh = int(parts[2]); fl = int(parts[3]); ch = int(parts[4])
                rest = parts[5:]
                pts = []
                # 优先解析“仅点序列”新格式
                if len(rest) >= 6 and len(rest) % 2 == 0:
                    for i in range(0, len(rest), 2):
                        x = float(rest[i]) * img_w
                        y = float(rest[i + 1]) * img_h
                        pts.append((x, y))
                    if len(pts) >= 3:
                        anns.append(Annotation(kind='polygon', points=pts, species=sp, cell_org=co, shape=sh, flagella=fl, chloroplast=ch))
                        continue
                # 兼容旧格式（含 bbox 四元组 + 点序列）
                if len(parts) >= 9:
                    xc = float(parts[5]); yc = float(parts[6]); ww = float(parts[7]); hh = float(parts[8])
                    rest2 = parts[9:]
                    pts2 = []
                    for i in range(0, len(rest2), 2):
                        x = float(rest2[i]) * img_w
                        y = float(rest2[i + 1]) * img_h
                        pts2.append((x, y))
                    if len(pts2) >= 3:
                        anns.append(Annotation(kind='polygon', points=pts2, species=sp, cell_org=co, shape=sh, flagella=fl, chloroplast=ch))
                    else:
                        x = (xc - ww / 2.0) * img_w
                        y = (yc - hh / 2.0) * img_h
                        anns.append(Annotation(kind='rect', rect=(x, y, ww * img_w, hh * img_h), species=sp, cell_org=co, shape=sh, flagella=fl, chloroplast=ch))
            except Exception:
                continue
    return anns


class Main(QMainWindow):
    def __init__(self, image_dir: Optional[str], output_dir: Optional[str], cfg_path: str):
        super().__init__()
        self.setWindowTitle('µSHM-YOLO 统一标注 GUI')
        self.cfg = Config(cfg_path)
        self.image_dir = image_dir or ''
        self.output_dir = output_dir or ''
        self.images = list_images(self.image_dir) if self.image_dir else []
        self.idx = 0
        self.current_img_stats: dict = {}
        self.annotations_cache = {}

        # UI
        self.canvas = Canvas()
        # 记录当前有焦点/打开的下拉框，用于数字键快速选择
        self.active_combo: Optional[QComboBox] = None
        right = QWidget()
        rlayout = QVBoxLayout(right)

        # mode & io buttons
        btn_polygon = QPushButton('多边形模式')
        btn_rect = QPushButton('矩形模式')
        btn_undo = QPushButton('撤销一点')
        btn_finish = QPushButton('完成多边形')
        btn_clear = QPushButton('清空全部')
        btn_choose_images = QPushButton('导入图片目录')
        btn_choose_output = QPushButton('选择输出目录')
        btn_auto_export = QPushButton('自动导出TXT')
        rlayout.addWidget(btn_polygon); rlayout.addWidget(btn_rect)
        rlayout.addWidget(btn_undo); rlayout.addWidget(btn_finish); rlayout.addWidget(btn_clear)
        rlayout.addWidget(btn_choose_images); rlayout.addWidget(btn_choose_output); rlayout.addWidget(btn_auto_export)

        # default class selectors
        rlayout.addWidget(QLabel('默认分类（用于新标注）：'))
        self.cb_species = QComboBox(); self.cb_cell = QComboBox(); self.cb_shape = QComboBox(); self.cb_flag = QComboBox(); self.cb_chl = QComboBox()
        rlayout.addWidget(self.cb_species); rlayout.addWidget(self.cb_cell); rlayout.addWidget(self.cb_shape); rlayout.addWidget(self.cb_flag); rlayout.addWidget(self.cb_chl)
        # 监听焦点，记录当前活跃下拉框
        for cb in (self.cb_species, self.cb_cell, self.cb_shape, self.cb_flag, self.cb_chl):
            cb.installEventFilter(self)

        # list and per-annotation editors
        rlayout.addWidget(QLabel('标注列表：'))
        self.list = QListWidget()
        rlayout.addWidget(self.list)
        self.btn_delete = QPushButton('删除选中')
        rlayout.addWidget(self.btn_delete)
        # per-annotation confirm (apply current class selectors)
        self.btn_apply_classes = QPushButton('确定（结束当前微藻）')
        rlayout.addWidget(self.btn_apply_classes)

        # nav and save
        btn_prev = QPushButton('上一张')
        btn_next = QPushButton('下一张')
        btn_save = QPushButton('导出TXT')
        rlayout.addWidget(btn_prev); rlayout.addWidget(btn_next); rlayout.addWidget(btn_save)

        # image list (右下角小窗口)
        rlayout.addWidget(QLabel('图片列表：'))
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(180)
        rlayout.addWidget(self.image_list)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        self.setCentralWidget(splitter)

        # signals
        btn_polygon.clicked.connect(lambda: self.canvas.set_mode('polygon'))
        btn_rect.clicked.connect(lambda: self.canvas.set_mode('rect'))
        btn_undo.clicked.connect(self.canvas.undo_point)
        btn_finish.clicked.connect(self.canvas.finish_polygon)
        btn_clear.clicked.connect(self.canvas.clear_all)

        btn_prev.clicked.connect(self.prev_image)
        btn_next.clicked.connect(self.next_image)
        btn_save.clicked.connect(self.save_current)
        btn_choose_images.clicked.connect(self.choose_images)
        btn_choose_output.clicked.connect(self.choose_output)
        btn_auto_export.clicked.connect(self.save_current)
        self.list.currentRowChanged.connect(self.on_select)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_apply_classes.clicked.connect(self.apply_classes_to_selected)
        self.canvas.selection_changed.connect(self.on_canvas_select)
        self.image_list.currentRowChanged.connect(self.on_image_choose)

        # init class options
        self.init_class_options()
        if self.images:
            self.populate_image_list()
            self.load_current()
        else:
            QMessageBox.information(self, '启动提示', '请点击左侧“导入图片目录”和“选择输出目录”后开始标注')

    def init_class_options(self):
        # group lists
        species_ids = self.cfg.groups.get('species', {}).get('classes', [])
        cell_ids = self.cfg.groups.get('cell_organization', self.cfg.groups.get('cell_org', {})).get('classes', [])
        # 兼容旧10/11显示（保存时不强制映射，训练期由 YAML 处理）
        if 10 not in cell_ids:
            cell_ids += [10]
        if 11 not in cell_ids:
            cell_ids += [11]
        shape_ids = self.cfg.groups.get('shape', {}).get('classes', [])
        flag_ids = self.cfg.groups.get('flagella', {}).get('classes', [])
        chl_ids = self.cfg.groups.get('chloroplast', {}).get('classes', [])

        # 保存完整可选集合，用于软规则筛选
        self.all_species_ids = list(species_ids)
        self.all_cell_ids = list(cell_ids)
        self.all_shape_ids = list(shape_ids)
        self.all_flag_ids = list(flag_ids)
        self.all_chl_ids = list(chl_ids)

        def fill_combo(combo: QComboBox, ids):
            combo.clear()
            for cid in ids:
                combo.addItem(f"{cid}: {self.cfg.id_name(cid)}", cid)

        fill_combo(self.cb_species, species_ids)
        fill_combo(self.cb_cell, cell_ids)
        fill_combo(self.cb_shape, shape_ids)
        fill_combo(self.cb_flag, flag_ids)
        fill_combo(self.cb_chl, chl_ids)

        # set defaults
        self.cb_species.setCurrentIndex(0)
        # 默认 cell_org 选第一个条目，通常是 7
        self.cb_cell.setCurrentIndex(0)
        self.cb_shape.setCurrentIndex(0)
        self.cb_flag.setCurrentIndex(0)
        self.cb_chl.setCurrentIndex(0)

        self.update_default_classes()
        # 物种变更触发软规则筛选 + 更新默认分类
        self.cb_species.currentIndexChanged.connect(self.on_species_changed)
        self.cb_cell.currentIndexChanged.connect(self.update_default_classes)
        self.cb_shape.currentIndexChanged.connect(self.update_default_classes)
        self.cb_flag.currentIndexChanged.connect(self.update_default_classes)
        self.cb_chl.currentIndexChanged.connect(self.update_default_classes)

    def on_species_changed(self, idx: int):
        sp = self.cb_species.itemData(idx)
        self.apply_soft_rule_filters(sp)
        self.update_default_classes()

    def update_default_classes(self):
        sp = self.cb_species.currentData()
        co = self.cb_cell.currentData()
        sh = self.cb_shape.currentData()
        fl = self.cb_flag.currentData()
        ch = self.cb_chl.currentData()
        self.canvas.set_default_classes((sp, co, sh, fl, ch))

    def apply_soft_rule_filters(self, species_id: Optional[int]):
        """根据 YAML 软规则实时筛选形状、鞭毛、群体组织的下拉选项。"""
        # 默认允许集合
        allow_shape = set(self.all_shape_ids)
        allow_flag = set(self.all_flag_ids)
        allow_cell = set(self.all_cell_ids)

        # 提取规则
        rules = self.cfg.cfg.get('stage3_integration', {}).get('rule_engine', {})
        if not rules.get('enabled', False):
            # 无规则时恢复完整集合
            self._refill_combo(self.cb_shape, self.all_shape_ids)
            self._refill_combo(self.cb_flag, self.all_flag_ids)
            self._refill_combo(self.cb_cell, self.all_cell_ids)
            return
        rule_list = rules.get('species_morphology_rules', [])
        # 疑似物种映射：31..37 → 0..6；38..43 不受软规则约束
        uncertain_map = {31: 0, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 6}
        if species_id in (38, 39, 40, 41, 42, 43):
            # 直接恢复完整集合并返回
            self._refill_combo(self.cb_shape, self.all_shape_ids)
            self._refill_combo(self.cb_flag, self.all_flag_ids)
            self._refill_combo(self.cb_cell, self.all_cell_ids)
            return
        species_for_rule = uncertain_map.get(species_id, species_id)
        # 查找该 species 的规则
        for r in rule_list:
            if r.get('species') != species_for_rule:
                continue
            req = r.get('required', {})
            forb = r.get('forbidden', {})
            # required 交集
            if 'shape' in req:
                allow_shape &= set(req['shape'])
            if 'flagella' in req:
                allow_flag &= set(req['flagella'])
            if 'cell_org' in req or 'cell_organization' in req:
                allow_cell &= set(req.get('cell_org', req.get('cell_organization', [])))
            # forbidden 排除
            if 'shape' in forb:
                allow_shape -= set(forb['shape'])
            if 'flagella' in forb:
                allow_flag -= set(forb['flagella'])
            if 'cell_org' in forb or 'cell_organization' in forb:
                allow_cell -= set(forb.get('cell_org', forb.get('cell_organization', [])))
            break

        # 若筛完为空，回退到完整集合，避免卡死
        if not allow_shape:
            allow_shape = set(self.all_shape_ids)
        if not allow_flag:
            allow_flag = set(self.all_flag_ids)
        if not allow_cell:
            allow_cell = set(self.all_cell_ids)

        # 重填下拉框并尽量保持当前选择（若仍合法）
        self._refill_combo(self.cb_shape, sorted(allow_shape))
        self._refill_combo(self.cb_flag, sorted(allow_flag))
        self._refill_combo(self.cb_cell, sorted(allow_cell))

    def _refill_combo(self, combo: QComboBox, ids):
        current_val = combo.currentData()
        combo.clear()
        for cid in ids:
            combo.addItem(f"{cid}: {self.cfg.id_name(cid)}", cid)
        # 保持原选择（若仍存在）
        if current_val in ids:
            for i in range(combo.count()):
                if combo.itemData(i) == current_val:
                    combo.setCurrentIndex(i)
                    break

    def set_combos_to_classes(self, sp: int, co: int, sh: int, fl: int, ch: int):
        def set_combo_to(combo: QComboBox, val: int):
            for i in range(combo.count()):
                if combo.itemData(i) == val:
                    combo.setCurrentIndex(i)
                    return
        set_combo_to(self.cb_species, sp)
        set_combo_to(self.cb_cell, co)
        set_combo_to(self.cb_shape, sh)
        set_combo_to(self.cb_flag, fl)
        set_combo_to(self.cb_chl, ch)

    def load_current(self):
        img_path = self.images[self.idx]
        img = read_image_any(img_path)
        if img is None:
            QMessageBox.warning(self, '警告', f'无法读取图像: {img_path}')
            return
        h, w = img.shape[:2]
        # 写入真实像素尺寸到 config.microscope.pixel_dimensions_px，便于头部注入
        self.cfg.microscope['pixel_dimensions_px'] = f"{w}x{h}"
        # 计算像素统计，用于TXT头部注入
        self.current_img_stats = compute_image_stats(img)
        self.canvas.set_image(img)
        # 加载已有标注
        if img_path in self.annotations_cache:
            # 使用缓存（未保存的临时标注）
            import copy
            self.canvas.annotations = copy.deepcopy(self.annotations_cache[img_path])
        else:
            self.canvas.annotations = load_labels_if_exist(self.cfg, img_path, self.output_dir, w, h)
        self.refresh_list()
        # 同步图片列表选择
        self.image_list.setCurrentRow(self.idx)

    def refresh_list(self):
        self.list.clear()
        for i, ann in enumerate(self.canvas.annotations):
            name = f"#{i} [{ann.kind}] sp={ann.species} co={ann.cell_org} sh={ann.shape} fl={ann.flagella} ch={ann.chloroplast}"
            self.list.addItem(QListWidgetItem(name))

    def on_select(self, idx: int):
        # 兼容列表刷新/删除后触发的旧索引，避免越界崩溃
        if idx < 0 or idx >= len(self.canvas.annotations):
            self.canvas.set_selected(None)
            return
        self.canvas.set_selected(idx)
        ann = self.canvas.annotations[idx]
        self.set_combos_to_classes(ann.species, ann.cell_org, ann.shape, ann.flagella, ann.chloroplast)

    def on_canvas_select(self, idx: int):
        # 画布选择联动列表与分类控件
        if 0 <= idx < len(self.canvas.annotations):
            self.list.setCurrentRow(idx)
            ann = self.canvas.annotations[idx]
            self.set_combos_to_classes(ann.species, ann.cell_org, ann.shape, ann.flagella, ann.chloroplast)
        else:
            # 无选中或越界时同步清空选择状态
            try:
                self.list.setCurrentRow(-1)
            except Exception:
                pass
            self.canvas.set_selected(None)

    def delete_selected(self):
        self.canvas.delete_selected()
        self.refresh_list()

    def prev_image(self):
        if not self.images:
            QMessageBox.information(self, '提示', '尚未选择图像目录')
            return
        self.store_current_to_cache()
        self.idx = (self.idx - 1) % len(self.images)
        self.load_current()

    def next_image(self):
        if not self.images:
            QMessageBox.information(self, '提示', '尚未选择图像目录')
            return
        self.store_current_to_cache()
        self.idx = (self.idx + 1) % len(self.images)
        self.load_current()

    def on_image_choose(self, row: int):
        if row < 0 or row >= len(self.images):
            return
        # 切换到选中图片（带缓存保存）
        self.store_current_to_cache()
        self.idx = row
        self.load_current()

    def populate_image_list(self):
        self.image_list.clear()
        for p in self.images:
            self.image_list.addItem(QListWidgetItem(os.path.basename(p)))

    def store_current_to_cache(self):
        if not self.images:
            return
        img_path = self.images[self.idx]
        import copy
        self.annotations_cache[img_path] = copy.deepcopy(self.canvas.annotations)

    def save_current(self):
        if not self.images:
            QMessageBox.information(self, '提示', '尚未选择图像目录')
            return
        if not self.output_dir:
            QMessageBox.information(self, '提示', '尚未设置输出目录')
            return
        img_path = self.images[self.idx]
        # 使用当前画布的图像尺寸，避免重复读取
        if self.canvas.image is None:
            img = read_image_any(img_path)
            if img is None:
                QMessageBox.warning(self, '警告', f'无法读取图像: {img_path}')
                return
            h, w = img.shape[:2]
        else:
            h, w = self.canvas.image.shape[:2]
        out = save_labels(self.cfg, img_path, self.output_dir, self.canvas.annotations, w, h, self.current_img_stats)
        QMessageBox.information(self, '保存成功', f'已保存到\n{out}')
        # 保存后写入缓存（保证回看仍显示当前标注）
        self.store_current_to_cache()

    # 键盘快捷键：Q=结束当前微藻，W=撤销一点，Space=下一张，Ctrl/Ctrl+S=导出TXT，数字1..9选择当前下拉框条目
    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key_Q:
            self.canvas.finish_polygon()
            return
        if key == Qt.Key_W:
            self.canvas.undo_point()
            return
        if key == Qt.Key_Space:
            self.next_image()
            return
        if key == Qt.Key_Control or (mods & Qt.ControlModifier and key in (Qt.Key_S, Qt.Key_Return, Qt.Key_Enter)):
            self.save_current()
            return
        # 数字键 1..9 选择当前活跃下拉框的第 n 项
        if Qt.Key_1 <= key <= Qt.Key_9 and self.active_combo is not None:
            idx = key - Qt.Key_1
            if 0 <= idx < self.active_combo.count():
                self.active_combo.setCurrentIndex(idx)
            return

        super().keyPressEvent(event)

    # 事件过滤器：记录当前活跃下拉框（获得焦点或弹出）
    def eventFilter(self, obj, ev):
        from PyQt5.QtCore import QEvent
        if isinstance(obj, QComboBox):
            if ev.type() in (QEvent.FocusIn, QEvent.Show, QEvent.MouseButtonPress):
                self.active_combo = obj
        return super().eventFilter(obj, ev)

    def choose_images(self):
        start_dir = self.image_dir if os.path.isdir(self.image_dir) else os.path.expanduser('~')
        d = QFileDialog.getExistingDirectory(self, '选择图像根目录（如 .../images/train）', start_dir)
        if d:
            self.image_dir = d
            self.images = list_images(self.image_dir)
            if not self.images:
                QMessageBox.warning(self, '警告', '所选目录未找到图像文件')
                return
            QMessageBox.information(self, '导入成功', f'已加载 {len(self.images)} 张图片')
            self.idx = 0
            self.annotations_cache.clear()
            self.populate_image_list()
            self.load_current()

    def choose_output(self):
        start_dir = self.output_dir if os.path.isdir(self.output_dir) else os.path.expanduser('~')
        d = QFileDialog.getExistingDirectory(self, '选择标注输出目录（如 .../labels/train）', start_dir)
        if d:
            self.output_dir = d
            os.makedirs(self.output_dir, exist_ok=True)

    def apply_classes_to_selected(self):
        sp = self.cb_species.currentData()
        co = self.cb_cell.currentData()
        sh = self.cb_shape.currentData()
        fl = self.cb_flag.currentData()
        ch = self.cb_chl.currentData()
        # 若仍在绘制多边形，按“确定”同时结束该微藻并应用分类
        if self.canvas.mode == 'polygon' and len(self.canvas.current_poly) >= 3:
            # 先完成多边形（默认分类），再对新标注应用当前分类
            self.canvas.finish_polygon()
            # 确保选中最新标注
            idx = len(self.canvas.annotations) - 1
            self.canvas.set_selected(idx)
            self.canvas.update_selected_classes(sp, co, sh, fl, ch)
        else:
            self.canvas.update_selected_classes(sp, co, sh, fl, ch)
        self.refresh_list()


def main():
    parser = argparse.ArgumentParser(description='µSHM-YOLO 统一标注 GUI')
    default_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov13_transformer_unified_v2_1.yaml')
    parser.add_argument('--image_dir', type=str, required=False, default=None, help='可选：图像根目录（如 .../images/train）')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='可选：标注输出目录（如 .../labels/train）')
    parser.add_argument('--cfg', type=str, default=default_cfg, help='统一 YAML 配置路径')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = Main(args.image_dir, args.output_dir, args.cfg)
    win.resize(1400, 860)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()