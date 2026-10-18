"""Evaluation protocol used for all reported numbers.

A change map is stored as an 8-bit grayscale PNG, binarised with Otsu's threshold on its
256-bin histogram (pixel >= t is "changed"), and compared with the reference map.
"""

import numpy as np


def otsu_u8(img):
    hist = np.bincount(img.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = weight_b = 0.0
    best_var, threshold = -1.0, 0
    for value in range(256):
        weight_b += hist[value]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += value * hist[value]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var, threshold = var_between, value
    return threshold


def binarize(gray_u8):
    t = otsu_u8(gray_u8)
    return (gray_u8 >= t).astype(np.uint8) * 255, t


def binary_metrics(ref, pred):
    """OA, Kappa, Precision, Recall and F1 (in %) of a binary prediction."""
    ref = ref > 0
    pred = pred > 0
    tp = int((ref & pred).sum())
    tn = int((~ref & ~pred).sum())
    fp = int((~ref & pred).sum())
    fn = int((ref & ~pred).sum())
    total = tp + tn + fp + fn
    oa = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    kappa = (oa - pe) / (1.0 - pe) if abs(1.0 - pe) > 1e-12 else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {k: v * 100.0 for k, v in
            dict(OA=oa, Kappa=kappa, Precision=precision, Recall=recall, F1=f1).items()}


def evaluate_gray(ref, gray_u8):
    pred, t = binarize(gray_u8)
    out = binary_metrics(ref, pred)
    out["threshold"] = int(t)
    return out, pred


def error_map(ref, pred):
    """RGB error map: white = TP, black = TN, red = FP, green = FN."""
    ref = ref > 0
    pred = pred > 0
    out = np.zeros((*ref.shape, 3), dtype=np.uint8)
    out[ref & pred] = (255, 255, 255)
    out[~ref & pred] = (255, 0, 0)
    out[ref & ~pred] = (0, 255, 0)
    return out
