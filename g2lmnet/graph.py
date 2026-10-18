"""Weighted region-adjacency graph (RAG) on co-superpixels and graph-prior inference."""

import math
from collections import defaultdict

import numpy as np
import torch

from .losses import normalize01


def build_region_graph(objects, img_t1, img_t2, labels_used, contrast_scale=0.25, pos_scale=2.0, min_weight=1e-4):
    """Edge weight = normalised shared-boundary length * exp(-a * feature contrast) * exp(-b * centroid distance)."""
    objects_np = objects.detach().cpu().numpy().astype(np.int64)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels_used)}
    n = len(labels_used)

    h, w = objects_np.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centroids = np.zeros((n, 2), dtype=np.float32)
    features = np.zeros((n, img_t1.shape[-1] * 2), dtype=np.float32)
    x1 = img_t1.detach().cpu().float().numpy()
    x2 = img_t2.detach().cpu().float().numpy()
    for lbl, idx in label_to_idx.items():
        mask = objects_np == lbl
        if not np.any(mask):
            continue
        centroids[idx] = yy[mask].mean(), xx[mask].mean()
        features[idx] = np.concatenate([x1[mask].mean(axis=0), x2[mask].mean(axis=0)])

    edge_counts = defaultdict(int)
    for a_map, b_map in ((objects_np[:, :-1], objects_np[:, 1:]), (objects_np[:-1, :], objects_np[1:, :])):
        boundary = a_map != b_map
        for a, b in zip(a_map[boundary].reshape(-1), b_map[boundary].reshape(-1)):
            ia, ib = label_to_idx.get(int(a)), label_to_idx.get(int(b))
            if ia is not None and ib is not None and ia != ib:
                edge_counts[(min(ia, ib), max(ia, ib))] += 1

    if not edge_counts:
        return (
            torch.empty((2, 0), dtype=torch.long, device=objects.device),
            torch.empty((0,), dtype=torch.float32, device=objects.device),
        )

    max_count = max(edge_counts.values())
    diag = math.sqrt(float(h * h + w * w)) + 1e-6
    contrast_den = features.std(axis=0).mean() + 1e-6

    src, dst, weights = [], [], []
    for (i, j), count in edge_counts.items():
        contrast = np.linalg.norm(features[i] - features[j]) / contrast_den
        pos_dist = np.linalg.norm(centroids[i] - centroids[j]) / diag
        weight = float((count / max_count) * np.exp(-contrast_scale * contrast) * np.exp(-pos_scale * pos_dist))
        weight = max(weight, min_weight)
        src.extend([i, j])
        dst.extend([j, i])
        weights.extend([weight, weight])

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=objects.device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=objects.device)
    return edge_index, edge_weight


def smooth_node_scores(scores, edge_index, edge_weight, alpha=0.35, num_iters=2):
    """d_i <- (d_i + alpha * sum_j w_ji d_j / sum_j w_ji) / (1 + alpha), repeated num_iters times."""
    if edge_index.numel() == 0 or alpha <= 0 or num_iters <= 0:
        return scores
    out = scores
    src, dst = edge_index
    n = scores.numel()
    for _ in range(num_iters):
        denom = torch.zeros(n, dtype=out.dtype, device=out.device)
        numer = torch.zeros(n, dtype=out.dtype, device=out.device)
        denom.index_add_(0, dst, edge_weight)
        numer.index_add_(0, dst, edge_weight * out[src])
        neigh = numer / (denom + 1e-6)
        out = (out + alpha * neigh) / (1.0 + alpha)
    return out


@torch.no_grad()
def compute_graph_prior(encoder, node_means_t1, node_means_t2, node_index, edge_index, edge_weight,
                        smooth_alpha=0.35, smooth_iters=2):
    """Returns the graph prior P_graph in [0, 1] and the graph-response weight C_graph in (0, 1), both (H, W).

    ``node_index`` is an (H, W) long tensor mapping every pixel to its superpixel node.
    """
    encoder.eval()
    _, z1 = encoder(node_means_t1)
    _, z2 = encoder(node_means_t2)
    diff = smooth_node_scores((z1 - z2).pow(2).mean(1), edge_index, edge_weight, smooth_alpha, smooth_iters)
    conf_sp = torch.sigmoid((diff - diff.mean()) / (diff.std() + 1e-6))

    diff_map = diff[node_index]
    conf_map = conf_sp[node_index]

    score_map = (diff_map - diff_map.mean()) / (diff_map.std() + 1e-6)
    prior = normalize01(score_map).clamp(0.0, 1.0)
    return prior, conf_map
