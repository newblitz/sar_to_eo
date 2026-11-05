"""Build pre-decoded image and frozen-VAE latent caches."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from diffusers import AutoencoderKL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sar2eo.fastdata import cache_latents, predecode, read_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--out", default="/kaggle/working/cache")
    parser.add_argument("--vae", default="assets/sd-vae-ft-ema")
    parser.add_argument("--root", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for split in ("train", "val", "test"):
        prefix = os.path.join(args.out, split)
        marker = prefix + ".done"
        if os.path.exists(marker):
            print(f"{split}: already cached, skipping")
            continue
        pairs = read_pairs(os.path.join(args.splits, f"{split}.txt"), args.root)
        print(f"{split}: {len(pairs)} pairs; first two: {pairs[:2]}", flush=True)
        predecode(pairs, prefix, args.workers)
        cache_latents(vae, prefix + "_eo.npy", prefix + "_lat.npy", device, args.batch)
        with open(marker, "w") as handle:
            handle.write(str(len(pairs)))
        print(f"{split}: done", flush=True)


if __name__ == "__main__":
    main()
