import re
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class _Attention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__(); self.heads = heads; self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape; q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return self.proj(y)


class _Block(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__(); self.norm1 = nn.LayerNorm(dim); self.attn = _Attention(dim, heads); self.norm2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SARJEPAViT(nn.Module):
    """Encoder-only compatible subset of the supplied SAR-JEPA ViT checkpoint.

    The checkpoint's iRPE lookup tables are reported and ignored in this compact
    fallback because the upstream LoMaR source is not attached. All compatible
    patch, attention, MLP, norm and position weights are loaded.
    """
    def __init__(self, depth=12, dim=768, heads=12):
        super().__init__(); self.embed_dim = dim; self.patch_embed = nn.Module(); self.patch_embed.proj = nn.Conv2d(1, dim, 16, 16); self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.pos_embed = nn.Parameter(torch.zeros(1, 197, dim)); self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)]); self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.patch_embed.proj(x).flatten(2).transpose(1, 2); pos = self.pos_embed[:, 1:]
        pos = F.interpolate(pos.transpose(1, 2).reshape(1, self.embed_dim, 14, 14), size=(16, 16), mode="bicubic", align_corners=False).flatten(2).transpose(1, 2)
        h = h + pos
        for block in self.blocks: h = block(h)
        return self.norm(h)


class _FallbackViT(nn.Module):
    def __init__(self, in_chans=1):
        super().__init__(); self.patch = nn.Conv2d(in_chans, 768, 16, 16); self.norm = nn.LayerNorm(768)
    def forward(self, x): return self.norm(self.patch(x).flatten(2).transpose(1, 2))


class SARViT(nn.Module):
    def __init__(self, in_chans: int = 1, img_size: int = 256):
        super().__init__(); self.embed_dim = 768
        try:
            try:
                lib = torch.library.Library("torchvision", "DEF")
                lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
                lib.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
            except Exception:
                pass
            import timm
            # The explicit SAR-JEPA checkpoint is loaded later; never trigger an
            # implicit network download while constructing the wrapper.
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="", class_token=False, img_size=img_size, in_chans=in_chans, drop_path_rate=0.1)
        except Exception:
            self.vit = _FallbackViT(in_chans)

    def forward(self, x):
        h = self.vit(x)
        if h.ndim == 4: h = h.flatten(2).transpose(1, 2)
        return h


def inspect_checkpoint(checkpoint: str | Path, report_path: str | Path = "results/sar_jepa_checkpoint_report.txt"):
    path = Path(checkpoint); state = torch.load(path, map_location="cpu", weights_only=False); model = state.get("model", state.get("state_dict", state)); keys = list(model)
    lines = [f"path: {path}", f"size_bytes: {path.stat().st_size}", f"top_level_keys: {list(state) if isinstance(state, dict) else type(state)}", f"model_entries: {len(model)}"]
    for key in ["module.patch_embed.proj.weight", "module.pos_embed", "module.cls_token", "module.blocks.0.attn.rpe_k.lookup_table_weight"]:
        if key in model: lines.append(f"{key}: {tuple(model[key].shape)}")
    lines.append("first_block:")
    lines.extend(f"{k}: {tuple(v.shape)}" for k, v in model.items() if k.startswith("module.blocks.0") and hasattr(v, "shape"))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True); Path(report_path).write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def load_sar_checkpoint(model: SARViT, checkpoint: str | None) -> None:
    if not checkpoint or not Path(checkpoint).exists(): print("SAR-JEPA checkpoint unavailable; using fallback:", checkpoint); return
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); raw = state.get("model", state.get("state_dict", state));
    has_rpe = any("rpe" in k for k in raw)
    if has_rpe:
        model.vit = SARJEPAViT()
    clean = {}
    for key, value in raw.items():
        if key.startswith(("module.decoder", "module.hogs", "module.mask_token", "module.encoder_pred")): continue
        key = key.removeprefix("module.").replace(".mlp.fc1.", ".mlp.0.").replace(".mlp.fc2.", ".mlp.2.")
        clean[key] = value
    missing, unexpected = model.vit.load_state_dict(clean, strict=False)
    print("SAR-JEPA iRPE tables present:", has_rpe)
    print("SAR checkpoint missing:", missing)
    print("SAR checkpoint unexpected:", unexpected)


def llrd_groups(vit: nn.Module, base_lr: float, decay: float = 0.75, wd: float = 0.05):
    groups = []; params = list(vit.named_parameters()); depth = max(1, len(params))
    for i, (name, param) in enumerate(params):
        if not param.requires_grad: continue
        no_decay = param.ndim == 1 or name.endswith("bias") or "norm" in name.lower()
        groups.append({"params": [param], "lr": base_lr * decay ** (depth - i - 1), "weight_decay": 0.0 if no_decay else wd})
    return groups
