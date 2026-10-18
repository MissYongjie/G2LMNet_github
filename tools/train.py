"""Train G2LMNet on one heterogeneous image pair (fully unsupervised, transductive).

Stage 1 pre-trains the KAN graph branch on co-superpixels and freezes it. Stage 2 trains the
LightMamba pixel branch under scheduled graph guidance. The reference map is read only to log
metrics; it is never used for training, early stopping or epoch selection. The reported result
is the change map of the last epoch.

Example:
    python tools/train.py --config configs/hetdata1.yaml --data_dir data/hetdata --work_dir work_dirs/hetdata1
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2lmnet.data import load_mat, pad_to_multiple, preprocess_pair, resolve_path, unpad  # noqa: E402
from g2lmnet.graph import build_region_graph, compute_graph_prior  # noqa: E402
from g2lmnet.losses import (  # noqa: E402
    EdgeAwareTVLoss,
    GradientAlignmentLoss,
    GraphLoss,
    PixelBaseLoss,
    edge_weight_from_images,
    make_pseudo_from_prior,
    normalize01,
)
from g2lmnet.metrics import error_map, evaluate_gray  # noqa: E402
from g2lmnet.models import KANDecoder, KANEncoder, LightMambaNet  # noqa: E402
from g2lmnet.superpixels import co_superpixels  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="G2LMNet unsupervised training")
    parser.add_argument("--config", required=True, help="configs/hetdataN.yaml")
    parser.add_argument("--data_dir", default="data/hetdata")
    parser.add_argument("--work_dir", default=None, help="default: work_dirs/<config name>")
    parser.add_argument("--seed", type=int, default=None, help="override the seed in the config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_all_epochs", action="store_true", help="save the prior/change map of every epoch")
    parser.add_argument("--no_ckpt", action="store_true", help="do not save the final weights")
    parser.add_argument("--cfg_options", nargs="+", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. pixel.lr=0.01 guidance.end_epoch=30")
    return parser.parse_args()


def load_config(path, options):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in options:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def set_seed(seed, deterministic):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def guidance_weight(epoch, g):
    """Global schedule w_g(e): 0 before E_start, linear ramp up to alpha_g, alpha_tail after E_end."""
    if epoch < g["start_epoch"]:
        return 0.0
    if epoch > g["end_epoch"]:
        return float(g["tail_strength"])
    if g["ramp_epochs"] <= 0:
        return float(g["strength"])
    return float(g["strength"] * min(1.0, (epoch - g["start_epoch"]) / max(1, g["ramp_epochs"])))


def pseudo_weight(epoch, g):
    """Pseudo-label weight w_p(e)."""
    if epoch < g["start_epoch"]:
        return 0.0
    if epoch > g["end_epoch"]:
        return float(g["pseudo_tail_weight"])
    return float(g["pseudo_weight"])


def to_u8(prob_map):
    return (prob_map.detach().cpu().numpy() * 255).astype(np.uint8)


def save_png(path, array):
    Image.fromarray(array).save(path)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.cfg_options)
    if args.seed is not None:
        cfg["seed"] = args.seed
    work_dir = args.work_dir or os.path.join("work_dirs", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)
    if args.save_all_epochs:
        os.makedirs(os.path.join(work_dir, "epochs"), exist_ok=True)
    with open(os.path.join(work_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device(args.device)
    set_seed(cfg["seed"], bool(cfg["deterministic"]))
    log_file = open(os.path.join(work_dir, "train_log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    sp_cfg, g_cfg, p_cfg, gd_cfg, l_cfg = (cfg[k] for k in ("superpixel", "graph", "pixel", "guidance", "loss"))
    epochs, pretrain_epochs = cfg["epochs"], g_cfg["pretrain_epochs"]
    assert epochs > pretrain_epochs, "epochs must exceed graph.pretrain_epochs"

    # ------------------------------------------------------------------ data
    mat_path = resolve_path(args.data_dir, cfg["dataset"]["file"])
    t1, t2, ref = load_mat(mat_path)
    t1, t2 = preprocess_pair(torch.from_numpy(t1).to(device), torch.from_numpy(t2).to(device),
                             cfg["dataset"]["t1_norm"], cfg["dataset"]["t2_norm"])
    img_t1, pad_h, pad_w = pad_to_multiple(t1, 32)
    img_t2, _, _ = pad_to_multiple(t2, 32)
    height, width, channels = img_t1.shape
    log(f"[Data] {mat_path}  size={ref.shape}  padded=({height}, {width})  channels={channels}")

    # ------------------------------------------------------------------ co-superpixels and RAG
    t0 = time.time()
    sp_path = os.path.join(work_dir, "superpixels.npy")
    if os.path.exists(sp_path):
        objects_np = np.load(sp_path)
    else:
        objects_np = co_superpixels(img_t1, img_t2, sp_cfg["method"], sp_cfg["n_segments"],
                                    sp_cfg["compactness"], sp_cfg["iterations"])
        np.save(sp_path, objects_np)
    objects = torch.from_numpy(objects_np).to(device)
    labels = torch.unique(objects).tolist()
    node_index = torch.zeros((height, width), dtype=torch.long, device=device)
    regions_t1, regions_t2 = [], []
    for i, lbl in enumerate(labels):
        mask = objects == lbl
        node_index[mask] = i
        regions_t1.append(img_t1[mask].reshape(-1, channels))
        regions_t2.append(img_t2[mask].reshape(-1, channels))
    node_means_t1 = torch.stack([r.mean(0) for r in regions_t1], dim=0)
    node_means_t2 = torch.stack([r.mean(0) for r in regions_t2], dim=0)
    edge_index, edge_weight = build_region_graph(objects, img_t1, img_t2, labels)
    log(f"[Graph] method={sp_cfg['method']} nodes={len(labels)} edges={edge_weight.numel() // 2} "
        f"({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------ models (construction order fixes the RNG stream)
    encoder = KANEncoder(channels, g_cfg["hidden_dim"], g_cfg["latent_dim"]).to(device)
    decoder_a = KANDecoder(g_cfg["latent_dim"], g_cfg["hidden_dim"], channels).to(device)
    decoder_b = KANDecoder(g_cfg["latent_dim"], g_cfg["hidden_dim"], channels).to(device)
    pixel_net = LightMambaNet(
        in_chans=channels,
        embed_dim=p_cfg["embed_dim"],
        kernel_size=p_cfg["alm_kernel"],
        num_blocks=p_cfg["alm_blocks"],
        expansion=p_cfg["alm_expansion"],
        dropout=p_cfg["alm_dropout"],
        aspp_rates=tuple(p_cfg["aspp_rates"]),
    ).to(device)

    graph_criterion = GraphLoss(lambda_reg=g_cfg["lambda_reg"])
    base_criterion = PixelBaseLoss(in_chans=channels, alpha=0.7, prior_beta=l_cfg["prior_beta"],
                                   aux_weight=l_cfg["aux_weight"])
    tv_criterion = EdgeAwareTVLoss()
    align_criterion = GradientAlignmentLoss()

    graph_params = list(encoder.parameters()) + list(decoder_a.parameters()) + list(decoder_b.parameters())
    graph_optimizer = AdamW(graph_params, lr=g_cfg["lr"], weight_decay=g_cfg["weight_decay"])
    pixel_optimizer = AdamW(pixel_net.parameters(), lr=p_cfg["lr"], weight_decay=p_cfg["weight_decay"])

    def graph_prior():
        return compute_graph_prior(encoder, node_means_t1, node_means_t2, node_index, edge_index, edge_weight,
                                   g_cfg["smooth_alpha"], g_cfg["smooth_iters"])

    # ------------------------------------------------------------------ stage 1: KAN graph branch
    for epoch in range(pretrain_epochs):
        encoder.train()
        decoder_a.train()
        decoder_b.train()
        total = 0.0
        for q1, q2 in zip(regions_t1, regions_t2):
            h1, z1 = encoder(q1)
            h2, z2 = encoder(q2)
            _, rec1 = decoder_a(z1)
            _, rec2 = decoder_b(z2)
            loss = graph_criterion(q1, q2, h1, z1, h2, z2, rec1, rec2, encoder, decoder_a, decoder_b)
            graph_optimizer.zero_grad()
            loss.backward()
            graph_optimizer.step()
            total += loss.item()
        prior, _ = graph_prior()
        prior_u8 = to_u8(unpad(prior, pad_h, pad_w))
        if args.save_all_epochs:
            save_png(os.path.join(work_dir, "epochs", f"graph_prior_e{epoch:03d}.png"), prior_u8)
        m, _ = evaluate_gray(ref, prior_u8)
        log(f"[Graph] epoch {epoch:3d}  loss={total / max(1, len(regions_t1)):.4f}  "
            f"prior OA={m['OA']:.2f} Kappa={m['Kappa']:.2f} F1={m['F1']:.2f}")

    # The KAN branch is frozen from here on: P_graph, C_graph and the pseudo labels are fixed.
    for p in graph_params:
        p.requires_grad_(False)
    prior, conf = graph_prior()
    prior_u8 = to_u8(unpad(prior, pad_h, pad_w))
    prior_metrics, prior_bin = evaluate_gray(ref, prior_u8)
    save_png(os.path.join(work_dir, "graph_prior.png"), prior_u8)
    save_png(os.path.join(work_dir, "graph_prior_error.png"), error_map(ref, prior_bin))

    x1 = img_t1.unsqueeze(0).permute(0, 3, 1, 2)
    x2 = img_t2.unsqueeze(0).permute(0, 3, 1, 2)
    prior_input = prior[None, None].clamp(0.0, 1.0)
    conf_input = conf[None, None].clamp(0.0, 1.0)
    pseudo, pseudo_map = make_pseudo_from_prior(prior, gd_cfg["pseudo_threshold"], gd_cfg["pseudo_blur_kernel"],
                                                gd_cfg["pseudo_blur_sigma"])
    w_edge = edge_weight_from_images(x1, x2, k=l_cfg["edge_k"])

    # ------------------------------------------------------------------ stage 2: LightMamba pixel branch
    final_u8 = None
    for epoch in range(pretrain_epochs, epochs):
        pixel_net.train()
        lr = p_cfg["lr"] * p_cfg["lr_gamma"] ** sum(epoch >= m_ for m_ in p_cfg["lr_milestones"])
        for group in pixel_optimizer.param_groups:
            group["lr"] = lr

        w_g = guidance_weight(epoch, gd_cfg)
        w_p = pseudo_weight(epoch, gd_cfg)
        x1_rec, x2_rec, prob, aux_probs = pixel_net(x1, x2, prior_input, gate_strength=w_g)
        base_loss, parts = base_criterion(x1_rec, x2_rec, x1, x2, prob, aux_probs, pseudo, pseudo_map,
                                          prior_input, prior_weight=w_g, pseudo_weight=w_p)

        if w_g > 0:  # adaptive graph-guided soft target, W = w_g * U_pix * C_graph
            with torch.no_grad():
                ambiguity = 1.0 - torch.abs(prob - 0.5) * 2.0
                w = w_g * ambiguity * conf_input
                target = (1.0 - w) * prob.detach() + w * prior_input
            fuse_loss = F.binary_cross_entropy(prob, target)
        else:
            fuse_loss = prob.new_tensor(0.0)
        tv_loss = tv_criterion(prob, w_edge)
        align_loss = align_criterion(prob, x1, x2)
        loss = (base_loss + l_cfg["lambda_fuse"] * fuse_loss + l_cfg["lambda_tv"] * tv_loss
                + l_cfg["lambda_align"] * align_loss)

        pixel_optimizer.zero_grad()
        loss.backward()
        pixel_optimizer.step()

        # Change map of this epoch's forward pass; the last one is the reported result.
        with torch.no_grad():
            final_u8 = to_u8(unpad(normalize01(prob[0, 0]), pad_h, pad_w))
        if args.save_all_epochs:
            save_png(os.path.join(work_dir, "epochs", f"change_map_e{epoch:03d}.png"), final_u8)
        m, _ = evaluate_gray(ref, final_u8)
        log(f"[Pixel] epoch {epoch:3d}  loss={loss.item():.4f}  w_g={w_g:.3f} w_p={w_p:.2f} lr={lr:.2e}  "
            f"OA={m['OA']:.2f} Kappa={m['Kappa']:.2f} F1={m['F1']:.2f}  "
            f"(rec={parts['rec']:.3f} pseudo={parts['pseudo']:.3f} fuse={fuse_loss.item():.3f} "
            f"tv={tv_loss.item():.4f} align={align_loss.item():.3f})")

    # ------------------------------------------------------------------ outputs
    final_metrics, final_bin = evaluate_gray(ref, final_u8)
    save_png(os.path.join(work_dir, "change_map.png"), final_u8)
    save_png(os.path.join(work_dir, "change_binary.png"), final_bin)
    save_png(os.path.join(work_dir, "change_error.png"), error_map(ref, final_bin))
    with open(os.path.join(work_dir, "metrics.json"), "w") as f:
        json.dump({"final_epoch": epochs - 1, "g2lmnet": final_metrics, "graph_prior": prior_metrics}, f, indent=2)
    if not args.no_ckpt:
        torch.save({"config": cfg, "pixel_net": pixel_net.state_dict(), "kan_encoder": encoder.state_dict(),
                    "kan_decoder_a": decoder_a.state_dict(), "kan_decoder_b": decoder_b.state_dict()},
                   os.path.join(work_dir, "checkpoint.pth"))
    log(f"[Done] final epoch {epochs - 1}: OA={final_metrics['OA']:.2f} Kappa={final_metrics['Kappa']:.2f} "
        f"F1={final_metrics['F1']:.2f}  (graph prior F1={prior_metrics['F1']:.2f})  -> {work_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
