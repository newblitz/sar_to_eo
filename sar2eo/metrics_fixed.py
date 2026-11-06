"""Corrected image metrics for SAR-to-EO evaluation.

All public metric inputs are uint8 arrays shaped ``(N, H, W, 3)``. Tensors
in the model's [-1, 1] range must be converted with :func:`m11_to_u8` once.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def m11_to_u8(x: torch.Tensor) -> np.ndarray:
    """Convert a BCHW tensor in [-1, 1] to uint8 BHWC, matching PNG encoding."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected tensor (N,3,H,W), got {tuple(x.shape)}")
    return (
        ((x.float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def _t01(u8: np.ndarray) -> torch.Tensor:
    if u8.dtype != np.uint8 or u8.ndim != 4 or u8.shape[-1] != 3:
        raise ValueError(f"expected uint8 (N,H,W,3), got {u8.dtype} {u8.shape}")
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float() / 255.0


def _check_pair(a_u8: np.ndarray, b_u8: np.ndarray) -> None:
    _t01(a_u8)
    _t01(b_u8)
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"shape mismatch {a_u8.shape} vs {b_u8.shape}")


def psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    """Per-image PSNR in dB using data_range=1 on [0, 1] images."""
    _check_pair(a_u8, b_u8)
    mse = ((_t01(a_u8) - _t01(b_u8)) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).numpy()


def _gauss(win: int, sigma: float) -> torch.Tensor:
    x = torch.arange(win).float() - (win - 1) / 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


def ssim(
    a_u8: np.ndarray,
    b_u8: np.ndarray,
    win: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> np.ndarray:
    """Per-image, RGB SSIM with a Gaussian window and data_range=1."""
    _check_pair(a_u8, b_u8)
    a, b = _t01(a_u8), _t01(b_u8)
    channels = a.shape[1]
    g = _gauss(win, sigma)
    kernel = (g[:, None] * g[None, :])[None, None].repeat(channels, 1, 1, 1)

    def filter_image(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, groups=channels)

    mu_a, mu_b = filter_image(a), filter_image(b)
    var_a = filter_image(a * a) - mu_a**2
    var_b = filter_image(b * b) - mu_b**2
    cov = filter_image(a * b) - mu_a * mu_b
    c1, c2 = k1**2, k2**2
    values = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return values.flatten(1).mean(1).numpy()


def edge_f1(
    gen_u8: np.ndarray,
    real_u8: np.ndarray,
    low: int = 100,
    high: int = 200,
    tol: int = 2,
) -> np.ndarray:
    """Per-image Canny edge F1 with symmetric pixel tolerance."""
    _check_pair(gen_u8, real_u8)
    import cv2

    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        if tol > 0
        else None
    )
    output = []
    for generated, real in zip(gen_u8, real_u8):
        generated_edges = cv2.Canny(cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY), low, high) > 0
        real_edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), low, high) > 0
        if not generated_edges.any() and not real_edges.any():
            output.append(1.0)
            continue
        generated_near_real = (
            cv2.dilate(generated_edges.astype(np.uint8), kernel) > 0
            if kernel is not None
            else generated_edges
        )
        real_near_generated = (
            cv2.dilate(real_edges.astype(np.uint8), kernel) > 0
            if kernel is not None
            else real_edges
        )
        precision = (generated_edges & real_near_generated).sum() / max(generated_edges.sum(), 1)
        recall = (real_edges & generated_near_real).sum() / max(real_edges.sum(), 1)
        output.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return np.asarray(output, dtype=np.float64)


class LPIPSMetric:
    """LPIPS AlexNet metric. Missing dependencies are reported as errors."""

    def __init__(self, device: str = "cpu"):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "lpips is not installed: run `pip install lpips`; refusing to skip LPIPS"
            ) from exc
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, a_u8: np.ndarray, b_u8: np.ndarray, batch: int = 32) -> np.ndarray:
        _check_pair(a_u8, b_u8)
        values = []
        for start in range(0, len(a_u8), batch):
            a = _t01(a_u8[start : start + batch]).to(self.device) * 2 - 1
            b = _t01(b_u8[start : start + batch]).to(self.device) * 2 - 1
            values.append(self.net(a, b).flatten().cpu())
        return torch.cat(values).numpy()


def fid_kid(
    real_u8: np.ndarray,
    fake_u8: np.ndarray,
    device: str = "cuda",
    kid_subset: int = 100,
) -> tuple[float, float, float]:
    """Compute FID and KID with torchmetrics and torch-fidelity."""
    _check_pair(real_u8, fake_u8)
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "torchmetrics[image] and torch-fidelity are required for FID/KID"
        ) from exc
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(
        subset_size=min(kid_subset, len(real_u8), len(fake_u8)), normalize=False
    ).to(device)
    for start in range(0, len(real_u8), 64):
        real = torch.from_numpy(np.ascontiguousarray(real_u8[start : start + 64])).permute(0, 3, 1, 2).to(device)
        fake = torch.from_numpy(np.ascontiguousarray(fake_u8[start : start + 64])).permute(0, 3, 1, 2).to(device)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        kid.update(real, real=True)
        kid.update(fake, real=False)
    kid_mean, kid_std = kid.compute()
    return float(fid.compute()), float(kid_mean), float(kid_std)


@torch.no_grad()
def vae_reconstruct(
    vae: torch.nn.Module,
    real_u8: np.ndarray,
    device: str = "cuda",
    batch: int = 32,
    scale: float = 0.18215,
) -> np.ndarray:
    """Reconstruct uint8 EO images through the frozen VAE."""
    _t01(real_u8)
    vae = vae.to(device).eval()
    output = []
    for start in range(0, len(real_u8), batch):
        x = _t01(real_u8[start : start + batch]).to(device) * 2 - 1
        z = vae.encode(x).latent_dist.mean * scale
        output.append(m11_to_u8(vae.decode(z / scale).sample))
    return np.concatenate(output)


def reference_candidates(real_u8: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Return aligned calibration image sets."""
    import cv2

    _t01(real_u8)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(real_u8))
    while len(real_u8) > 1 and np.array_equal(permutation, np.arange(len(real_u8))):
        permutation = rng.permutation(len(real_u8))
    mean_image = np.broadcast_to(
        real_u8.reshape(-1, 3).mean(0), real_u8.shape
    ).round().astype(np.uint8)
    return {
        "identity": real_u8,
        "blur_sigma2": np.stack([cv2.GaussianBlur(image, (0, 0), 2) for image in real_u8]),
        "brightness_+0.1": np.clip(real_u8.astype(np.int32) + 26, 0, 255).astype(np.uint8),
        "shuffled_pairs": real_u8[permutation],
        "dataset_mean_color": np.ascontiguousarray(mean_image),
    }


def per_image_metrics(
    gen_u8: np.ndarray,
    real_u8: np.ndarray,
    lpips_metric: LPIPSMetric | None = None,
    edge_tol: int = 2,
) -> dict[str, np.ndarray]:
    _check_pair(gen_u8, real_u8)
    values = {
        "psnr": psnr(gen_u8, real_u8),
        "ssim": ssim(gen_u8, real_u8),
        "edge_f1": edge_f1(gen_u8, real_u8, tol=edge_tol),
        "edge_f1_exact": edge_f1(gen_u8, real_u8, tol=0),
    }
    if lpips_metric is not None:
        values["lpips"] = lpips_metric(gen_u8, real_u8)
    return values


def summarize(per_image: dict[str, np.ndarray], terrain: list[str] | np.ndarray | None = None) -> dict:
    result = {"all": {key: float(np.mean(value)) for key, value in per_image.items()}}
    if terrain is not None:
        terrain_array = np.asarray(terrain)
        for category in sorted(set(terrain_array.tolist())):
            mask = terrain_array == category
            result[category] = {
                key: float(np.mean(value[mask])) for key, value in per_image.items()
            }
    return result


def color_stats(u8: np.ndarray) -> dict[str, list[float]]:
    _t01(u8)
    values = u8.reshape(-1, 3).astype(np.float64) / 255.0
    return {"mean": values.mean(0).round(4).tolist(), "std": values.std(0).round(4).tolist()}


def terrain_from_paths(split_file: str) -> list[str]:
    import re

    terrain = []
    for line in open(split_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = next(
            part
            for part in re.split(r"[\t,;\s]+", line)
            if part.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        parts = path.replace("\\", "/").split("/")
        marker = "s1" if "s1" in parts else "s2"
        terrain.append(parts[parts.index(marker) - 1])
    return terrain
