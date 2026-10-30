import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def _upstream_sar_jepa_encoder():
    """Build the official LoMaR/iRPE encoder from the cloned SAR-JEPA repo."""
    source_root = Path(__file__).resolve().parent.parent / "assets" / "SAR-JEPA" / "Pretraining"
    if not source_root.exists():
        raise FileNotFoundError(
            f"Official SAR-JEPA source is required for the iRPE checkpoint: {source_root}"
        )
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    from models_lomar import mae_vit_base_patch16_dec512d8b

    return mae_vit_base_patch16_dec512d8b(img_size=224, in_chans=1)


class SARJEPAViT(nn.Module):
    """Official SAR-JEPA/LoMaR encoder-only forward path with iRPE preserved."""
    def __init__(self):
        super().__init__()
        self.encoder = _upstream_sar_jepa_encoder()
        self.embed_dim = self.encoder.patch_embed.proj.out_channels
        if self.embed_dim != 768 or len(self.encoder.blocks) != 12:
            raise RuntimeError(
                "SAR-JEPA factory is not the required ViT-B/16: "
                f"width={self.embed_dim}, depth={len(self.encoder.blocks)}"
            )
    def forward(self, x):
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        h = self.encoder.patch_embed(x).float()
        cls = self.encoder.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls, h), dim=1)
        h = h + self.encoder.pos_embed
        for block in self.encoder.blocks:
            h = block(h)
        h = self.encoder.norm(h)[:, 1:]
        h = h.transpose(1, 2).reshape(h.shape[0], self.embed_dim, 14, 14)
        return F.interpolate(h, size=(16, 16), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)


class SARViT(nn.Module):
    """Required SAR-JEPA ViT-B/16 encoder; no generic-encoder fallback exists."""
    def __init__(self, in_chans: int = 1, img_size: int = 256):
        super().__init__()
        if in_chans != 1 or img_size != 256:
            raise ValueError("SAR-JEPA wrapper requires single-channel 256px SAR inputs.")
        self.embed_dim = 768
        self.vit = SARJEPAViT()

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
    """Load the required SAR-JEPA encoder and reject incomplete checkpoints."""
    if not checkpoint:
        raise FileNotFoundError("No SAR-JEPA checkpoint path was configured.")
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Required SAR-JEPA checkpoint does not exist: {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)
    raw = state.get("model", state.get("state_dict", state))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"SAR-JEPA checkpoint has no usable model state dictionary: {path}")
    clean = {}
    renamed_buffers = 0
    for key, value in raw.items():
        key = key.removeprefix("module.")
        # Older SAR-JEPA checkpoints called these fixed handcrafted feature
        # modules ``hogs*``; the current upstream factory calls them
        # ``sarfeature*``. They are non-trainable buffers, so this is a
        # verified state-dict naming compatibility alias, not a dropped key.
        for index in range(1, 5):
            old_prefix = f"hogs{index}."
            if key.startswith(old_prefix):
                key = f"sarfeature{index}." + key[len(old_prefix):]
                renamed_buffers += 1
                break
        clean[key] = value

    expected = set(model.vit.encoder.state_dict())
    received = set(clean)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    if missing or unexpected:
        raise RuntimeError(
            "SAR-JEPA checkpoint/model key mismatch. "
            f"missing={missing}; unexpected={unexpected}. "
            "The complete upstream SAR-JEPA model state must match."
        )
    try:
        incompatible = model.vit.encoder.load_state_dict(clean, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"SAR-JEPA checkpoint shape mismatch for {path}: {exc}") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Unexpected strict load result: {incompatible}")

    patch_key = "patch_embed.proj.weight"
    if not torch.equal(model.vit.encoder.state_dict()[patch_key].cpu(), clean[patch_key].cpu()):
        raise RuntimeError("SAR-JEPA patch embedding weight was not loaded from the checkpoint.")
    rpe_keys = [key for key in clean if ".rpe_" in key]
    if not rpe_keys:
        raise RuntimeError("SAR-JEPA checkpoint is missing required iRPE tables.")
    print(f"SAR-JEPA checkpoint loaded strictly: {path}")
    print(f"SAR-JEPA parameters loaded: {len(clean)}")
    if renamed_buffers:
        print(f"SAR-JEPA compatibility aliases applied: {renamed_buffers}")
    print(f"SAR-JEPA iRPE tables loaded: {len(rpe_keys)}")


def llrd_groups(vit: nn.Module, base_lr: float, decay: float = 0.75, wd: float = 0.05):
    groups = []; params = list(vit.named_parameters()); depth = max(1, len(params))
    for i, (name, param) in enumerate(params):
        if not param.requires_grad: continue
        no_decay = param.ndim == 1 or name.endswith("bias") or "norm" in name.lower()
        groups.append({"params": [param], "lr": base_lr * decay ** (depth - i - 1), "weight_decay": 0.0 if no_decay else wd})
    return groups
