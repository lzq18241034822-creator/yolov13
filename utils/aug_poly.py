# -*- coding: utf-8 -*-
# Polygon-safe augmentations for YOLOv13 seg training
import cv2
import numpy as np
from typing import List, Tuple

def letterbox_with_segments(img, segs: List[np.ndarray], new_shape=(640, 640), color=(114,114,114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])

    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    h0, w0 = shape
    h1, w1 = img.shape[:2]
    out_segs = []
    for s in segs:
        if s.size == 0:
            out_segs.append(s); continue
        p = s.copy()
        p[:, 0] *= w0; p[:, 1] *= h0
        p[:, 0] = p[:, 0] * ratio[0] + left
        p[:, 1] = p[:, 1] * ratio[1] + top
        p[:, 0] /= w1; p[:, 1] /= h1
        out_segs.append(p)
    return img, out_segs, ratio, (dw, dh)

def random_affine_with_segments(img, segs: List[np.ndarray], degrees=10, translate=0.1, scale=0.5, shear=10):
    h, w = img.shape[:2]
    R = degrees * (np.random.rand() - 0.5) * 2
    S = 1 + (np.random.rand() * 2 - 1) * scale
    T_x = translate * (np.random.rand() * 2 - 1) * w
    T_y = translate * (np.random.rand() * 2 - 1) * h
    Sh = shear * (np.random.rand() * 2 - 1)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), R, S)
    M[:, 2] += [T_x, T_y]
    M_shear = np.eye(3); M_shear[0, 1] = np.tan(np.radians(Sh))
    M_affine = M_shear[:2, :] @ np.vstack([M, [0,0,1]])

    img = cv2.warpAffine(img, M_affine, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114,114,114))

    out_segs = []
    for s in segs:
        if s.size == 0:
            out_segs.append(s); continue
        p = s.copy()
        p[:, 0] *= w; p[:, 1] *= h
        ones = np.ones((p.shape[0], 1), dtype=p.dtype)
        homo = np.hstack([p, ones])  # (K,3)
        p2 = (M_affine @ homo.T).T
        p2[:, 0] = (p2[:, 0] / w).clip(0,1)
        p2[:, 1] = (p2[:, 1] / h).clip(0,1)
        out_segs.append(p2)
    return img, out_segs

def hflip_with_segments(img, segs: List[np.ndarray]):
    img2 = cv2.flip(img, 1)
    out = []
    for s in segs:
        p = s.copy(); p[:, 0] = 1.0 - p[:, 0]; out.append(p)
    return img2, out

def vflip_with_segments(img, segs: List[np.ndarray]):
    img2 = cv2.flip(img, 0)
    out = []
    for s in segs:
        p = s.copy(); p[:, 1] = 1.0 - p[:, 1]; out.append(p)
    return img2, out