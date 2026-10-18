"""Loading and preprocessing of heterogeneous image pairs stored as .mat files."""

import os

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

# Older file names used by public HetData releases, accepted as fall-backs.
LEGACY_NAMES = {
    "hetdata1_gloucester1.mat": "dataset#2.mat",
    "hetdata2_texas.mat": "TexasALI.mat",
    "hetdata3_gloucester2.mat": "dataset#6.mat",
    "hetdata4_sardinia.mat": "dataset#1.mat",
    "hetdata5_toulouse.mat": "dataset#3.mat",
    "hetdata6_shuguang.mat": "dataset#4.mat",
}


def resolve_path(data_dir, file_name):
    for name in (file_name, LEGACY_NAMES.get(file_name)):
        if name and os.path.exists(os.path.join(data_dir, name)):
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"{file_name} not found in {data_dir} (see data/README.md)")


def load_mat(path):
    """Returns image_t1, image_t2 as float32 (H, W, C) arrays in [0, 1] and the binary GT (uint8 0/255)."""
    data = scipy.io.loadmat(path)
    t1 = data["image_t1"].astype(np.float32) / 255.0
    t2 = data["image_t2"].astype(np.float32) / 255.0
    ref = (data["Ref_gt"] > 0).astype(np.uint8) * 255
    return t1, t2, ref


def minmax(img):
    lo, hi = torch.min(img), torch.max(img)
    return (img - lo) / (hi - lo) if hi - lo != 0 else (img - lo) / (hi - lo + 1e-10)


def channel_zscore(img):
    h, w, c = img.shape
    flat = img.contiguous().view(-1, c)
    center = flat - flat.mean(0, keepdim=True)
    std = torch.sqrt((center ** 2).sum(0, keepdim=True) / (h * w))
    return (center / std).view(h, w, c)


def log_zscore(img):
    img = img.clone()
    img[torch.abs(img) <= 0] = torch.min(img[torch.abs(img) > 0])
    return channel_zscore(torch.log(img + 1.0))


def preprocess_pair(t1, t2, t1_mode, t2_mode):
    """Per-image normalisation: 'minmax' (global), 'zscore' (per channel) or 'log_zscore' (log(x+1), then zscore)."""
    fns = {"minmax": minmax, "zscore": channel_zscore, "log_zscore": log_zscore}
    if (t1_mode == "minmax") != (t2_mode == "minmax"):
        raise ValueError("'minmax' must be used for both images or for neither")
    return fns[t1_mode](t1), fns[t2_mode](t2)


def pad_to_multiple(img, multiple=32):
    """Reflect-pads an (H, W, C) tensor at the bottom/right so H and W are multiples of `multiple`."""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    img = F.pad(img.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode="reflect").permute(1, 2, 0)
    return img, pad_h, pad_w


def unpad(img, pad_h, pad_w):
    return img[: img.shape[0] - pad_h, : img.shape[1] - pad_w]
