"""LightMamba pixel branch: U-Net encoder-decoder with Axial LiteMamba (ALM) blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class AxialLiteMamba(nn.Module):
    """ALM block: GN -> (horizontal + vertical depthwise conv) * pointwise gate -> FFN, residual.

    A light, dependency-free alternative to a selective state-space scan.
    """

    def __init__(self, dim, kernel_size=7, expansion=4, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(1, dim)
        self.row_scan = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, pad), groups=dim)
        self.col_scan = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=dim)
        self.gate = nn.Conv2d(dim, dim, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, kernel_size=1),
        )
        self.drop = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = (self.row_scan(y) + self.col_scan(y)) * torch.sigmoid(self.gate(y))
        return x + self.drop(self.ffn(y))


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
                for r in rates
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c * (len(rates) + 1), out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        outs.append(F.interpolate(self.image_pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))


class ChangeGate(nn.Module):
    """Feature modulation by the graph prior: V * (1 + w_g * sigmoid(conv1x1(P_graph)))."""

    def __init__(self, channels):
        super().__init__()
        self.prior_proj = nn.Sequential(nn.Conv2d(1, channels, kernel_size=1), nn.Sigmoid())

    def forward(self, feat, prior, strength=1.0):
        if prior is None or strength <= 0:
            return feat
        prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat * (1.0 + strength * self.prior_proj(prior))


class DiffInject(nn.Module):
    """Resizes and projects |X2 - X1| to a given scale."""

    def __init__(self, in_chans, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, diff, size):
        return self.proj(F.interpolate(diff, size=size, mode="bilinear", align_corners=False))


class LightMambaNet(nn.Module):
    """Pixel branch of G2LMNet.

    Shared encoder for both dates, DiffInject at every scale, ALM blocks and ASPP at the
    bottleneck, ChangeGate on bottleneck and skip features, two reconstruction heads, one
    change head and two auxiliary heads.
    """

    def __init__(self, in_chans=3, embed_dim=32, kernel_size=7, num_blocks=2, expansion=4, dropout=0.0,
                 aspp_rates=(1, 2, 4, 8)):
        super().__init__()
        c = embed_dim
        self.down = nn.MaxPool2d(2)

        self.enc1 = conv_block(in_chans, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)

        blocks = [conv_block(c * 4, c * 8)]
        for _ in range(max(1, int(num_blocks))):
            blocks.append(AxialLiteMamba(c * 8, kernel_size=kernel_size, expansion=expansion, dropout=dropout))
        self.bottleneck = nn.Sequential(*blocks)
        self.aspp = ASPP(c * 8, c * 8, rates=aspp_rates)

        self.diff1 = DiffInject(in_chans, c)
        self.diff2 = DiffInject(in_chans, c * 2)
        self.diff3 = DiffInject(in_chans, c * 4)
        self.diffb = DiffInject(in_chans, c * 8)

        self.gate1 = ChangeGate(c)
        self.gate2 = ChangeGate(c * 2)
        self.gate3 = ChangeGate(c * 4)
        self.gateb = ChangeGate(c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.recon_head_1 = nn.Conv2d(c, in_chans, kernel_size=1)
        self.recon_head_2 = nn.Conv2d(c, in_chans, kernel_size=1)
        self.change_head = nn.Conv2d(c, 1, kernel_size=1)
        self.aux_head3 = nn.Conv2d(c * 4, 1, kernel_size=1)
        self.aux_head2 = nn.Conv2d(c * 2, 1, kernel_size=1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))
        return e1, e2, e3, b

    def forward(self, x1, x2, prior=None, gate_strength=1.0):
        e11, e12, e13, b1 = self._encode(x1)
        e21, e22, e23, b2 = self._encode(x2)
        diff = torch.abs(x2 - x1)

        skip1 = self.gate1(e11 + e21 + self.diff1(diff, e11.shape[-2:]), prior, gate_strength)
        skip2 = self.gate2(e12 + e22 + self.diff2(diff, e12.shape[-2:]), prior, gate_strength)
        skip3 = self.gate3(e13 + e23 + self.diff3(diff, e13.shape[-2:]), prior, gate_strength)

        b = self.aspp(b1 + b2 + self.diffb(diff, b1.shape[-2:]))
        b = self.gateb(b, prior, gate_strength)

        size = x1.shape[-2:]
        d3 = self.dec3(torch.cat([self.up3(b), skip3], dim=1))
        aux3 = torch.sigmoid(F.interpolate(self.aux_head3(d3), size=size, mode="bilinear", align_corners=False))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        aux2 = torch.sigmoid(F.interpolate(self.aux_head2(d2), size=size, mode="bilinear", align_corners=False))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))

        x1_recon = self.recon_head_1(d1)
        x2_recon = self.recon_head_2(d1)
        change_prob = torch.sigmoid(self.change_head(d1)).clamp(1e-6, 1.0 - 1e-6)
        return x1_recon, x2_recon, change_prob, [aux2, aux3]
