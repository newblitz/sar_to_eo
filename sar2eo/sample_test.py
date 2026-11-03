"""Run one CUDA-only SAR-to-EO training step."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sar_encoder import load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sar_encoder import load_sar_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Optional saved Config JSON.")
    parser.add_argument("--split", default="splits/train.txt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this test intentionally does not fall back to CPU.")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {args.device} is invalid; "
            f"available devices: {torch.cuda.device_count()}"
        )
    cfg = Config.load(args.config) if args.config else Config()
    split = Path(args.split)
    if not split.exists():
        raise FileNotFoundError(f"Missing split file: {split}")
    dataset = PairedSEN12(split, augment=False, image_size=cfg.image_size)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Sample index {args.index} is outside split of length {len(dataset)}")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    batch = dataset[args.index]
    sar = batch["sar"].unsqueeze(0).to(device)
    eo = batch["eo"].unsqueeze(0).to(device)
    model = SAR2EO(
        cfg.d_model,
        cfg.depth,
        cfg.heads,
        cfg.align_layer,
        cfg.dino_dim,
        cfg.in_chans,
    )
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    model = model.to(device).train()
    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_new, betas=(0.9, 0.95))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            x0, dino = targets.encode(eo, cfg.vae_scale)
        t = torch.rand(x0.shape[0], device=device)
        eps = torch.randn_like(x0)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
        velocity, aligned = model(
            xt,
            t,
            sar,
            torch.zeros(x0.shape[0], dtype=torch.bool, device=device),
            sar_trainable=True,
        )
        flow_loss = F.mse_loss(velocity.float(), (eps - x0).float())
        align_loss = -(
            F.normalize(aligned.float(), dim=-1) * F.normalize(dino.float(), dim=-1)
        ).sum(-1).mean()
        loss = flow_loss + cfg.alignment_lambda * align_loss

    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    if not any(
        gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("No finite gradients were produced for trainable parameters.")
    optimizer.step()

    print(
        {
            "status": "PASS",
            "device": torch.cuda.get_device_name(args.device),
            "sar": tuple(sar.shape),
            "eo": tuple(eo.shape),
            "latent": tuple(x0.shape),
            "dino": tuple(dino.shape),
            "velocity": tuple(velocity.shape),
            "aligned": tuple(aligned.shape),
            "loss": float(loss.detach().item()),
            "optimizer_step": "completed",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
