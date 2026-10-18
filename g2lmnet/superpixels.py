"""Shared co-superpixel partition of an image pair.

* ``slico``     : OpenCV SLICO on the "normdiff" guidance image [||X1||, ||X2||, ||X2 - X1||].
* ``soft_slic`` : a non-learning soft-SLIC (SSN-style soft assignment) computed on X2.
"""

import math

import cv2 as cv
import numpy as np
import torch


def _to_hwc_float(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    if x.ndim == 2:
        x = x[..., None]
    return x.astype(np.float32)


def _minmax01(a, eps=1e-6):
    return (a - a.min()) / (a.max() - a.min() + eps)


def _robust_stretch01(a, p=1.0):
    lo, hi = np.percentile(a, p), np.percentile(a, 100 - p)
    if hi <= lo:
        return _minmax01(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _to_uint8_3ch(img):
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _relabel_contiguous(labels):
    _, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int32)


def slico_normdiff(img_t1, img_t2, n_segments, compactness=10.0, num_iterations=10):
    x1, x2 = _to_hwc_float(img_t1), _to_hwc_float(img_t2)
    h, w = x1.shape[:2]
    n1 = np.sqrt((x1 ** 2).sum(2, keepdims=True) + 1e-8)
    n2 = np.sqrt((x2 ** 2).sum(2, keepdims=True) + 1e-8)
    nd = np.sqrt(((x2 - x1) ** 2).sum(2, keepdims=True) + 1e-8)
    guide = np.concatenate([n1, n2, nd], 2)
    for c in range(3):
        guide[..., c] = _robust_stretch01(_minmax01(guide[..., c]))
    region_size = max(8, int(math.sqrt((h * w) / max(1, n_segments))))
    sp = cv.ximgproc.createSuperpixelSLIC(
        _to_uint8_3ch(guide), algorithm=cv.ximgproc.SLICO, region_size=region_size, ruler=float(compactness)
    )
    sp.iterate(num_iterations)
    return _relabel_contiguous(sp.getLabels())


def soft_slic(img, n_segments, iters=10, compactness=0.12, tau=1.0):
    """Soft SLIC: local soft assignments exp(-D/tau) with D = ||f - c_k||^2 + (m/S)^2 ||p - p_k||^2."""
    X = _to_hwc_float(img)
    H, W, C = X.shape
    s = max(4.0, math.sqrt((H * W) / max(1, n_segments)))
    R = int(2 * s)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    ny = max(1, int(round(H / s)))
    nx = max(1, int(round(W / s)))
    CY, CX = np.meshgrid(
        np.linspace(s / 2, H - s / 2, ny).astype(np.float32),
        np.linspace(s / 2, W - s / 2, nx).astype(np.float32),
        indexing="ij",
    )
    centers_xy = np.stack([CY.reshape(-1), CX.reshape(-1)], axis=1)[:n_segments]
    K0 = centers_xy.shape[0]
    cyi = np.clip(np.round(centers_xy[:, 0]).astype(int), 0, H - 1)
    cxi = np.clip(np.round(centers_xy[:, 1]).astype(int), 0, W - 1)
    centers_f = X[cyi, cxi, :].copy()

    def window(k):
        cy, cx = centers_xy[k]
        y0, y1 = max(0, int(cy - R)), min(H, int(cy + R) + 1)
        x0, x1 = max(0, int(cx - R)), min(W, int(cx + R) + 1)
        if y1 <= y0 or x1 <= x0:
            return None
        df = X[y0:y1, x0:x1, :] - centers_f[k][None, None, :]
        dy, dx = yy[y0:y1, x0:x1] - cy, xx[y0:y1, x0:x1] - cx
        D = (df * df).sum(2) + (compactness / s) ** 2 * (dy * dy + dx * dx)
        return (slice(y0, y1), slice(x0, x1)), np.exp(-D / max(1e-8, tau))

    def normaliser():
        denom = np.zeros((H, W), dtype=np.float32) + 1e-8
        for k in range(K0):
            win = window(k)
            if win is not None:
                denom[win[0]] += win[1]
        return denom

    for _ in range(iters):
        denom = normaliser()
        sumw = np.zeros((K0,), dtype=np.float64)
        sumf = np.zeros((K0, C), dtype=np.float64)
        sumy = np.zeros((K0,), dtype=np.float64)
        sumx = np.zeros((K0,), dtype=np.float64)
        for k in range(K0):
            win = window(k)
            if win is None:
                continue
            sl, a = win
            w64 = (a / denom[sl]).astype(np.float64)
            sumw[k] += w64.sum()
            sumf[k] += (w64[..., None] * X[sl]).reshape(-1, C).sum(0)
            sumy[k] += (w64 * yy[sl]).sum()
            sumx[k] += (w64 * xx[sl]).sum()
        valid = sumw > 1e-8
        centers_f[valid] = (sumf[valid] / sumw[valid][:, None]).astype(np.float32)
        centers_xy[valid, 0] = (sumy[valid] / sumw[valid]).astype(np.float32)
        centers_xy[valid, 1] = (sumx[valid] / sumw[valid]).astype(np.float32)

    denom = normaliser()
    labels = -np.ones((H, W), dtype=np.int32)
    best = -np.ones((H, W), dtype=np.float32)
    for k in range(K0):
        win = window(k)
        if win is None:
            continue
        sl, a = win
        a = a / denom[sl]
        take = a > best[sl]
        best[sl][take] = a[take]
        labels[sl][take] = k
    return _relabel_contiguous(labels)


def co_superpixels(img_t1, img_t2, method, n_segments, compactness, num_iterations=10):
    if method == "slico":
        return slico_normdiff(img_t1, img_t2, n_segments, compactness, num_iterations)
    if method == "soft_slic":
        return soft_slic(img_t2, n_segments, iters=num_iterations, compactness=compactness)
    raise ValueError(f"Unknown superpixel method: {method}")
