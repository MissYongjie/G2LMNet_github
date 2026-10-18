"""KAN graph branch: residual KAN encoder and two KAN decoders on superpixel nodes."""

import torch.nn as nn

from .kan import KANLinear


class KANBlock(nn.Module):
    """Two KANLinear layers with LayerNorm, GELU, dropout and a residual path."""

    def __init__(self, in_ch, hid_ch, out_ch, p_drop=0.1):
        super().__init__()
        self.lin1 = KANLinear(in_ch, hid_ch)
        self.lin2 = KANLinear(hid_ch, out_ch)
        self.norm_in = nn.LayerNorm(in_ch)
        self.norm_mid = nn.LayerNorm(hid_ch)
        self.norm_out = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.res_proj = nn.Identity() if in_ch == out_ch else nn.Linear(in_ch, out_ch)

    def forward(self, x):
        h = self.lin1(self.norm_in(x))
        h = self.drop(self.act(self.norm_mid(h)))
        y = self.drop(self.norm_out(self.lin2(h)))
        return self.res_proj(x) + y


class KANEncoder(nn.Module):
    """Returns the penultimate block output H and the latent embedding Z."""

    def __init__(self, in_channels, hidden_channels, out_channels, depth=3, p_drop=0.1):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (depth - 1) + [out_channels]
        self.blocks = nn.ModuleList(
            [KANBlock(dims[i], max(dims[i], dims[i + 1]), dims[i + 1], p_drop) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats[-2] if len(feats) >= 2 else feats[-1], feats[-1]


class KANDecoder(nn.Module):
    """Two residual KAN blocks; returns the hidden feature and the reconstruction."""

    def __init__(self, in_channels, hidden_channels, out_channels, p_drop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                KANBlock(in_channels, max(in_channels, hidden_channels), hidden_channels, p_drop),
                KANBlock(hidden_channels, max(hidden_channels, out_channels), out_channels, p_drop),
            ]
        )

    def forward(self, x):
        h = self.blocks[0](x)
        return h, self.blocks[1](h)
