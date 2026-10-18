"""Training objectives of G2LMNet (graph pre-training loss and pixel-branch losses)."""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM


def normalize01(t, eps=1e-6):
    if t.dim() == 2:
        return (t - t.amin()) / (t.amax() - t.amin() + eps)
    if t.dim() == 4:
        tmin = t.amin(dim=(-2, -1), keepdim=True)
        tmax = t.amax(dim=(-2, -1), keepdim=True)
        return (t - tmin) / (tmax - tmin + eps)
    raise ValueError(f"Unsupported shape for normalize01: {tuple(t.shape)}")


# --------------------------------------------------------------------------- graph branch
class GraphLoss(nn.Module):
    """L_graph = L_rec + L_con + lambda_reg * ||Theta_KAN||^2 (per superpixel region)."""

    def __init__(self, lambda_reg=1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg

    def l2_regularization(self, module):
        device = next(module.parameters()).device
        reg = torch.zeros((), device=device)
        for param in module.parameters():
            reg = reg + (param ** 2).sum()
        return self.lambda_reg * reg

    def forward(self, q1, q2, h1, z1, h2, z2, rec1, rec2, encoder, decoder_a, decoder_b):
        rec = F.mse_loss(rec1, q1) + F.mse_loss(rec2, q2)
        con = F.mse_loss(h1, h2) + F.mse_loss(z1, z2)
        reg = (
            self.l2_regularization(decoder_b)
            + self.l2_regularization(decoder_a)
            + self.l2_regularization(encoder)
        )
        return rec + con + reg


# --------------------------------------------------------------------------- pixel branch
def make_pseudo_from_prior(prior_map, tau_high=0.7, blur_kernel=33, sigma=3.0):
    """Graph-derived pseudo labels: Y~ = 1[GaussianBlur(sigmoid(zscore(P_graph))) > tau].

    Returns the binary pseudo labels and the blurred map M used as BCE spatial weights.
    """
    diff_np = prior_map.detach().cpu().float().numpy()
    z_map = (diff_np - diff_np.mean()) / (diff_np.std() + 1e-6)
    norm_map = 1.0 / (1.0 + np.exp(-z_map))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_map = cv2.GaussianBlur(norm_map, (blur_kernel, blur_kernel), sigma)
    pseudo = (blur_map > tau_high).astype(np.float32)
    pseudo = torch.from_numpy(pseudo)[None, None].to(prior_map.device)
    weight = torch.from_numpy(blur_map.astype(np.float32))[None, None].to(prior_map.device)
    return pseudo, weight


def dice_loss(prob, target, smooth=1e-6):
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    intersection = (prob * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (prob.sum() + target.sum() + smooth)


def finite_grad_2d(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def edge_weight_from_images(x1, x2, k=5.0):
    """W_edge = exp(-k * minmax(|grad| of the channel-mean inputs)) on interior pixels."""
    g1x, g1y = finite_grad_2d(x1.mean(1, keepdim=True))
    g2x, g2y = finite_grad_2d(x2.mean(1, keepdim=True))
    gx = (torch.abs(g1x) + torch.abs(g2x)) / 2.0
    gy = (torch.abs(g1y) + torch.abs(g2y)) / 2.0
    mag = torch.sqrt(gx[..., :-1, :] ** 2 + gy[..., :, :-1] ** 2 + 1e-8)
    return torch.exp(-k * normalize01(mag))


class EdgeAwareTVLoss(nn.Module):
    def forward(self, p, w):
        gx, gy = finite_grad_2d(p)
        return (w * (torch.abs(gx[..., :-1, :]) + torch.abs(gy[..., :, :-1]))).mean()


class GradientAlignmentLoss(nn.Module):
    """1 - cos between gradients of P and of the channel-mean |X2 - X1|, where |grad D| > 1e-3."""

    def forward(self, p, x1, x2):
        gx_p, gy_p = finite_grad_2d(p)
        gx_d, gy_d = finite_grad_2d(torch.abs(x2 - x1).mean(1, keepdim=True))
        gx_p, gy_p = gx_p[..., :-1, :], gy_p[..., :, :-1]
        gx_d, gy_d = gx_d[..., :-1, :], gy_d[..., :, :-1]
        dot = gx_p * gx_d + gy_p * gy_d
        mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + 1e-8)
        mag_d = torch.sqrt(gx_d ** 2 + gy_d ** 2 + 1e-8)
        cos = dot / (mag_p * mag_d + 1e-8)
        mask = (mag_d > 1e-3).float()
        return (mask * (1.0 - cos)).mean()


class PixelBaseLoss(nn.Module):
    """L_base = L_rec^pix + w_p(e) (L_pseudo + lambda_a L_aux) + beta w_g(e) MSE(P_pix, P_graph)."""

    def __init__(self, in_chans=3, alpha=0.7, prior_beta=0.3, aux_weight=0.2):
        super().__init__()
        self.alpha = alpha
        self.prior_beta = prior_beta
        self.aux_weight = aux_weight
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=in_chans)
        self.l1 = nn.L1Loss()

    @staticmethod
    def _pseudo_loss(prob, pseudo, weight):
        prob = prob.clamp(1e-6, 1.0 - 1e-6)
        bce = (F.binary_cross_entropy(prob, pseudo, reduction="none") * weight).mean()
        return bce + dice_loss(prob, pseudo)

    def forward(self, x1_recon, x2_recon, x1, x2, change_prob, aux_probs, pseudo, pseudo_weight_map,
                prior_map, prior_weight, pseudo_weight):
        ssim_loss = (1.0 - self.ssim(x1_recon, x1)) + (1.0 - self.ssim(x2_recon, x2))
        l1_loss = self.l1(x1_recon, x1) + self.l1(x2_recon, x2)
        recon = self.alpha * ssim_loss + (1.0 - self.alpha) * l1_loss

        change = self._pseudo_loss(change_prob, pseudo, pseudo_weight_map)
        aux = 0.0
        for aux_prob in aux_probs:
            aux = aux + self._pseudo_loss(aux_prob, pseudo, pseudo_weight_map)
        aux = self.aux_weight * aux

        prior = self.prior_beta * prior_weight * F.mse_loss(change_prob, prior_map)
        total = recon + pseudo_weight * (change + aux) + prior
        return total, {
            "rec": float(recon.detach()),
            "pseudo": float((pseudo_weight * change).detach()),
            "aux": float(pseudo_weight * torch.as_tensor(aux).detach()),
            "prior": float(prior.detach()),
        }
