import torch
from torch import nn
from pathlib import Path
import sys
try:
    from .sar_encoder import SARViT
    from .compat import ensure_torchvision_ops
except ImportError:
    from sar2eo.sar_encoder import SARViT
    from sar2eo.compat import ensure_torchvision_ops


class SiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.cond = nn.Linear(dim, dim)

    def forward(self, h, c):
        h = h + self.cond(c).unsqueeze(1)
        q = self.norm1(h); h = h + self.attn(q, q, q, need_weights=False)[0]
        return h + self.mlp(self.norm2(h))


class SAR2EO(nn.Module):
    """SiT-shaped latent flow transformer with SAR token conditioning."""
    def __init__(self, d_model=768, depth=12, heads=12, align_layer=4, dino_dim=768, in_chans=1):
        super().__init__(); self.d_model, self.depth, self.align_layer = d_model, depth, align_layer
        self.sar_encoder = SARViT(in_chans)
        self.cond_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.cond_proj[-1].weight); nn.init.zeros_(self.cond_proj[-1].bias)
        self.glob_proj = nn.Linear(768, d_model); nn.init.zeros_(self.glob_proj.weight); nn.init.zeros_(self.glob_proj.bias)
        self._uses_local_sit = False
        if d_model == 768 and depth == 12 and heads == 12:
            ensure_torchvision_ops()
            sit_root = Path(__file__).resolve().parent.parent / "assets" / "SiT"
            if sit_root.exists():
                sys.path.insert(0, str(sit_root))
                from models import SiT_B_2
                self.backbone = SiT_B_2(input_size=32, in_channels=4, learn_sigma=False)
                self.x_embedder = self.backbone.x_embedder; self.pos_embed = self.backbone.pos_embed; self.t_embedder = self.backbone.t_embedder; self.blocks = self.backbone.blocks; self.final_layer = self.backbone.final_layer; self._uses_local_sit = True
        if not self._uses_local_sit:
            self.x_embedder = nn.Conv2d(4, d_model, kernel_size=2, stride=2)
            self.pos_embed = nn.Parameter(torch.zeros(1, 256, d_model))
            self.t_embedder = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.blocks = nn.ModuleList([SiTBlock(d_model, heads) for _ in range(depth)])
        self.inject = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(depth)])
        for layer in self.inject: nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        if not self._uses_local_sit: self.final_layer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 16))
        self.align_proj = nn.Sequential(nn.Linear(d_model, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, dino_dim))
        self.null_tok = nn.Parameter(torch.zeros(1, 256, d_model)); self.null_glob = nn.Parameter(torch.zeros(1, d_model))

    def encode_sar(self, sar):
        tokens = self.sar_encoder(sar)
        return self.cond_proj(tokens), self.glob_proj(tokens.mean(1))

    def denoise(self, x_t, t, cond, g, drop=None):
        b = x_t.shape[0]
        if drop is not None:
            mask = drop.bool().view(b, 1, 1)
            cond = torch.where(mask, self.null_tok.expand(b, -1, -1), cond)
            g = torch.where(mask.squeeze(-1), self.null_glob.expand(b, -1), g)
        c = self.t_embedder(t.float().view(b)) + g
        h = self.x_embedder(x_t)
        if h.ndim == 4: h = h.flatten(2).transpose(1, 2)
        h = h + self.pos_embed + cond
        aligned = None
        for i, block in enumerate(self.blocks):
            h = h + self.inject[i](cond)
            h = block(h, c)
            if i + 1 == self.align_layer: aligned = self.align_proj(h)
        if self._uses_local_sit:
            velocity = self.final_layer(h, c)
            velocity = self.backbone.unpatchify(velocity)
        else:
            patch = self.final_layer(h).transpose(1, 2).reshape(b, 16, 16, 16)
            velocity = patch.reshape(b, 4, 4, 16, 16).permute(0, 1, 3, 2, 4).reshape(b, 4, 32, 32)
        return velocity, aligned if self.training else None

    def forward(self, x_t, t, sar, drop=None, sar_trainable=True):
        with torch.set_grad_enabled(sar_trainable): cond, g = self.encode_sar(sar)
        return self.denoise(x_t, t, cond, g, drop)

