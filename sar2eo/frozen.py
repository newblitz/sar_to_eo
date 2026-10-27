"""Frozen SD-VAE and DINOv2 targets used by training and the VAE ceiling check."""
import torch
from torch import nn
import torch.nn.functional as F


_TV_LIB = None


def _torchvision_compat():
    # The host's CPU torchvision wheel omits compiled NMS ops while its meta
    # registration assumes they exist. Defining the schemas keeps optional
    # torchvision imports used by Transformers/Diffusers functional.
    global _TV_LIB
    try:
        _TV_LIB = torch.library.Library("torchvision", "DEF")
        _TV_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        _TV_LIB.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


class FrozenTargets(nn.Module):
    def __init__(self, vae_id: str, dino_id: str):
        super().__init__()
        _torchvision_compat()
        from diffusers import AutoencoderKL
        from transformers import Dinov2Model
        self.vae = AutoencoderKL.from_pretrained(vae_id, local_files_only=True)
        self.dino = Dinov2Model.from_pretrained(dino_id, local_files_only=True)
        self.vae.eval().requires_grad_(False); self.dino.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, eo: torch.Tensor, scale: float = 0.18215):
        z = self.vae.encode(eo).latent_dist.sample() * scale
        rgb = (eo + 1) / 2
        rgb = F.interpolate(rgb, (224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=eo.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=eo.device)[None, :, None, None]
        tokens = self.dino(pixel_values=(rgb - mean) / std).last_hidden_state[:, 1:]
        if tokens.shape[1] != 256: raise ValueError(f"Expected 256 DINO tokens, got {tokens.shape}")
        return z, tokens

    @torch.no_grad()
    def decode(self, z: torch.Tensor, scale: float = 0.18215):
        return self.vae.decode(z / scale).sample.clamp(-1, 1)
