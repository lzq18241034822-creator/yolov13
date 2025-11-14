# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# 尝试使用 PIL 以支持中文文本绘制
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

def color_palette(i: int):
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    return colors[i % len(colors)]

def _try_load_cn_font(size: int = 16) -> Tuple[bool, Any]:
    """尝试加载中文字体，返回 (ok, font)。"""
    if not PIL_AVAILABLE:
        return False, None
    # 常见 Windows 字体路径（优先微软雅黑）
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
    ]
    for p in candidates:
        try:
            f = ImageFont.truetype(p, size=size)
            return True, f
        except Exception:
            continue
    # 其他平台的常见 CJK 字体
    candidates_linux = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates_linux:
        try:
            f = ImageFont.truetype(p, size=size)
            return True, f
        except Exception:
            continue
    return False, None

def _draw_text_cn(out_bgr: np.ndarray, text: str, org: Tuple[int,int], color: Tuple[int,int,int]):
    """在 BGR 图像上绘制中文文本，支持描边；若 PIL 不可用则退化到 cv2.putText。"""
    if not text:
        return out_bgr
    if PIL_AVAILABLE:
        ok, font = _try_load_cn_font(size=18)
        if ok and font is not None:
            # 转 PIL RGB
            img_pil = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            # 文本描边提高可读性
            stroke_fill = (0, 0, 0)
            fill = (int(color[2]), int(color[1]), int(color[0]))  # BGR->RGB
            draw.text((org[0], org[1]), text, font=font, fill=fill, stroke_width=2, stroke_fill=stroke_fill)
            return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    # 退化：替换非 ASCII 字符避免显示为问号
    safe = ''.join(ch if ord(ch) < 128 else '·' for ch in text)
    cv2.putText(out_bgr, safe, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out_bgr

def draw_instance_overlay(img: np.ndarray,
                          masks: List[np.ndarray],
                          rois: List[Dict[str,Any]],
                          labels: List[Dict[str,Any]],
                          rule_infos: List[Dict[str,Any]]):
    out = img.copy()
    H, W = out.shape[:2]
    # 为每个实例半透明填充颜色，提高可见性
    for i, m in enumerate(masks):
        c = color_palette(i)
        m_uint = (m.astype(np.uint8) > 0)
        if m_uint.any():
            alpha = 0.25
            color_arr = np.array([[c[0], c[1], c[2]]], dtype=np.uint8)
            # 逐像素融合
            out[m_uint] = (out[m_uint] * (1 - alpha) + color_arr * alpha).astype(np.uint8)

    # 绘制轮廓、框与文本
    for i, m in enumerate(masks):
        c = color_palette(i)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, cnts, -1, c, 2)
        ys, xs = np.where(m>0)
        x0=x1=y0=y1=None
        if len(xs)>0:
            x0,x1 = int(xs.min()), int(xs.max())
            y0,y1 = int(ys.min()), int(ys.max())
            x0 = max(0, x0); y0 = max(0, y0); x1 = min(W-1, x1); y1 = min(H-1, y1)
            cv2.rectangle(out, (x0,y0), (x1,y1), c, 2)

        # 文本（标签与降级提示），锚定到 bbox 左上角
        txt = None
        if i < len(labels):
            lab = labels[i]
            sp = lab.get('species_name', 'sp'); co = lab.get('cell_org_name', 'co'); sh = lab.get('shape_name', 'sh')
            fl = lab.get('flagella_name', 'fl'); ch = lab.get('chloroplast_name', 'ch')
            txt = f"{sp}|{co}|{sh}|{fl}|{ch}"
        if i < len(rule_infos) and not rule_infos[i].get('passed', True):
            txt = (txt or '') + " [demoted]"
        if txt:
            xt = 10 if x0 is None else max(0, x0)
            yt = 20 if y0 is None else max(12, y0 - 6)
            out = _draw_text_cn(out, txt, (xt, yt), c)
    return out