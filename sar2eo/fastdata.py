"""Cached uint8 image and frozen-VAE latent data path for fast training."""

from __future__ import annotations

import os
import re
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SIZE = 256
LATENT = SIZE // 8
VAE_SCALE = 0.18215
VARIANTS = [(), (3,), (2,), (2, 3)]
NP_AXES = [(), (1,), (0,), (0, 1)]


def _is(path: str, tag: str) -> bool:
    path = path.replace("\\", "/")
    return f"_{tag}_" in os.path.basename(path) or f"/{tag}/" in path


def read_pairs(manifest: str, root: str | None = None) -> list[tuple[str, str]]:
    pairs = []
    with open(manifest) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [
                value
                for value in re.split(r"[\t,;\s]+", line)
                if value.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(fields) < 2:
                raise ValueError(f"cannot find two image paths in row: {line!r}")
            sar, eo = fields[:2]
            if _is(sar, "s2") and _is(eo, "s1"):
                sar, eo = eo, sar
            if root:
                sar = sar if os.path.isabs(sar) else os.path.join(root, sar)
                eo = eo if os.path.isabs(eo) else os.path.join(root, eo)
            pairs.append((sar, eo))
    return pairs


def _load(pair: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    sar_path, eo_path = pair
    sar = Image.open(sar_path).convert("L")
    eo = Image.open(eo_path).convert("RGB")
    if sar.size != (SIZE, SIZE):
        sar = sar.resize((SIZE, SIZE), Image.Resampling.BOX)
    if eo.size != (SIZE, SIZE):
        eo = eo.resize((SIZE, SIZE), Image.Resampling.BOX)
    return np.asarray(sar, dtype=np.uint8), np.asarray(eo, dtype=np.uint8)


def predecode(
    pairs: list[tuple[str, str]], out_prefix: str, workers: int = 4
) -> None:
    sar = np.lib.format.open_memmap(
        out_prefix + "_sar.npy", mode="w+", dtype=np.uint8, shape=(len(pairs), SIZE, SIZE)
    )
    eo = np.lib.format.open_memmap(
        out_prefix + "_eo.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(pairs), SIZE, SIZE, 3),
    )
    with Pool(max(1, workers)) as pool:
        for index, (sar_image, eo_image) in enumerate(
            pool.imap(_load, pairs, chunksize=16)
        ):
            sar[index], eo[index] = sar_image, eo_image
            if index % 1000 == 0:
                print(f"{out_prefix}: {index}/{len(pairs)}", flush=True)
    sar.flush()
    eo.flush()


@torch.no_grad()
def cache_latents(vae, eo_path: str, out_path: str, device="cuda", batch: int = 32):
    eo = np.load(eo_path, mmap_mode="r")
    latents = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float16,
        shape=(4, len(eo), 4, LATENT, LATENT),
    )
    vae = vae.to(device).eval()
    use_amp = str(device).startswith("cuda")
    for start in range(0, len(eo), batch):
        images = torch.from_numpy(np.array(eo[start : start + batch])).to(device)
        images = images.permute(0, 3, 1, 2).float() / 127.5 - 1
        for variant, dims in enumerate(VARIANTS):
            image = images.flip(dims) if dims else images
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                latent = vae.encode(image).latent_dist.mean * VAE_SCALE
            if not torch.isfinite(latent).all():
                raise RuntimeError("non-finite VAE latents; retry this batch in fp32")
            latents[variant, start : start + batch] = (
                latent.float().cpu().numpy().astype(np.float16)
            )
    latents.flush()


class FastPairs(Dataset):
    """Read cached images and matching cached latents without decoding PNGs."""

    def __init__(self, prefix: str, latents_path: str, train: bool = True):
        self.sar = np.load(prefix + "_sar.npy", mmap_mode="r")
        self.eo = np.load(prefix + "_eo.npy", mmap_mode="r")
        self.lat = np.load(latents_path, mmap_mode="r")
        if not len(self.sar) == len(self.eo) == self.lat.shape[1]:
            raise ValueError("cache files have different lengths")
        self.train = train

    def __len__(self):
        return len(self.sar)

    def __getitem__(self, index):
        variant = int(torch.randint(4, (1,))) if self.train else 0
        axes = NP_AXES[variant]
        sar, eo = np.array(self.sar[index]), np.array(self.eo[index])
        if axes:
            sar, eo = np.flip(sar, axes), np.flip(eo, axes)
        return (
            torch.from_numpy(np.ascontiguousarray(sar))[None],
            torch.from_numpy(np.ascontiguousarray(eo.transpose(2, 0, 1))),
            torch.from_numpy(np.array(self.lat[variant, index])),
        )


def to_model_inputs(sar_u8, eo_u8, latent_f16, device):
    sar = sar_u8.to(device, non_blocking=True).float() / 127.5 - 1
    eo = eo_u8.to(device, non_blocking=True).float() / 127.5 - 1
    return sar, eo, latent_f16.to(device, non_blocking=True).float()
