"""Generate and persist EO images once, independently of metric computation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sar2eo.configs import Config
from sar2eo.dataset import PairedSEN12
from sar2eo.metrics_fixed import m11_to_u8
from sar2eo.model import SAR2EO
from sar2eo.sample import sample
from sar2eo.sar_encoder import load_sar_checkpoint


def _cfg_name(value: float) -> str:
    return str(float(value))


def resolve_checkpoint(requested: str) -> Path:
    """Resolve common Kaggle upload names without guessing unrelated files."""
    path = Path(requested)
    candidates = [path]
    if path.suffix == "":
        candidates.extend((path.with_suffix(".pt"), path.with_suffix(".pth")))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) == 1:
        return existing[0]
    if path.is_dir():
        raise FileNotFoundError(
            f"checkpoint path is a directory, not a torch checkpoint file: {path}. "
            "Pass the uploaded best_ema.pt or best_ema.pth file, not its containing folder."
        )
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"EMA checkpoint was not found. Tried: {tried}. "
        "Check the Kaggle input filename with `find /kaggle/input -name 'best_ema*'`."
    )


def _save_grid(split: str, sar: np.ndarray, generated: np.ndarray, real: np.ndarray) -> None:
    panels = []
    for index in range(min(8, len(generated))):
        sar_rgb = np.repeat(sar[index][..., None], 3, axis=2)
        panels.append(np.concatenate((sar_rgb, generated[index], real[index]), axis=1))
    grid = np.concatenate(panels, axis=0)
    Image.fromarray(grid, mode="RGB").save(f"results/check_grid_{split}.png")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="runs/main/best_ema.pt")
    parser.add_argument("--config")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    if args.batch_size < 16:
        raise ValueError("--batch-size must be at least 16")
    if args.num_workers > 2:
        raise ValueError("--num-workers must be at most 2")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {config_path}")
    config = Config.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("ema")
    if state is None:
        raise ValueError(f"{checkpoint_path} has no EMA state; refusing to use raw weights")

    model = SAR2EO(config.d_model, config.depth, config.heads, config.align_layer, config.dino_dim)
    load_sar_checkpoint(model.sar_encoder, config.sar_checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(config.vae_id, local_files_only=True).to(device).eval()
    dataset = PairedSEN12(Path(config.split_root) / f"{args.split}.txt", augment=False, image_size=256)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generated, sar_images, real_images = [], [], []
    for batch in loader:
        latent = sample(model, batch["sar"].to(device, non_blocking=True), args.steps, args.cfg)
        decoded = vae.decode(latent / config.vae_scale).sample
        generated.append(m11_to_u8(decoded))
        sar_images.append(((batch["sar"] + 1) * 127.5).clamp(0, 255).byte().squeeze(1).numpy())
        real_images.append(m11_to_u8(batch["eo"]))

    generated_array = np.concatenate(generated)
    sar_array = np.concatenate(sar_images)
    real_array = np.concatenate(real_images)
    if len(generated_array) != len(dataset):
        raise AssertionError(f"generated {len(generated_array)} rows for {len(dataset)} split rows")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = _cfg_name(args.cfg)
    stem = output_dir / f"gen_{args.split}_cfg{cfg_name}_steps{args.steps}"
    np.save(stem.with_suffix(".npy"), generated_array)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint.get("step", 0)),
        "weights": "ema",
        "split": args.split,
        "count": len(dataset),
        "seed": args.seed,
        "cfg": args.cfg,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "vae_scale": config.vae_scale,
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    _save_grid(args.split, sar_array, generated_array, real_array)
    print(f"saved {stem}.npy ({generated_array.shape}, {generated_array.dtype})")


if __name__ == "__main__":
    main()
