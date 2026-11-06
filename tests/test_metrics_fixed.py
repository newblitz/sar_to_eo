import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sar2eo import metrics_fixed as M


def make_images(n=8, size=256, seed=0):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n):
        base = rng.random((16, 16, 3)).astype(np.float32)
        image = (cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC) * 200 + 20).clip(0, 255).astype(np.uint8)
        for _ in range(8):
            x, y = (int(value) for value in rng.integers(5, size - 80, 2))
            w, h = (int(value) for value in rng.integers(20, 70, 2))
            cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        images.append(np.clip(image.astype(np.float32) + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8))
    return np.stack(images)


def test_psnr_known_values_and_range_bug():
    base = np.full((2, 64, 64, 3), 100, np.uint8)
    shifted = np.full_like(base, 126)
    assert np.allclose(M.psnr(shifted, base), 20 * np.log10(255 / 26), atol=1e-4)
    assert (M.psnr(base, base) > 99).all()
    a, b = torch.rand(2, 3, 8, 8) * 2 - 1, torch.rand(2, 3, 8, 8) * 2 - 1
    mse = ((a - b) ** 2).flatten(1).mean(1)
    right = 10 * torch.log10(4.0 / mse)
    assert np.allclose(M.psnr(M.m11_to_u8(a), M.m11_to_u8(b)), right.numpy(), atol=0.1)


def test_m11_roundtrip():
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    back = M._t01(M.m11_to_u8(x)) * 2 - 1
    assert (back - x).abs().max() <= 1 / 127.5 + 1e-6
    assert M.m11_to_u8(torch.full((1, 3, 4, 4), 5.0)).max() == 255


def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity

    images = make_images()
    noisy = np.clip(images.astype(np.float32) + np.random.default_rng(1).normal(0, 12, images.shape), 0, 255).astype(np.uint8)
    expected = [
        structural_similarity(noisy_image / 255.0, image / 255.0, data_range=1.0, channel_axis=-1,
                              gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)
        for noisy_image, image in zip(noisy, images)
    ]
    assert np.allclose(M.ssim(noisy, images), expected, atol=2e-3)


def test_edge_f1_behaviour():
    images = make_images()
    assert np.allclose(M.edge_f1(images, images), 1.0)
    shifted = np.roll(images, 1, axis=2)
    assert (M.edge_f1(shifted, images, tol=2) > 0.85).all()
    assert (M.edge_f1(shifted, images, tol=0) < M.edge_f1(shifted, images, tol=2)).all()
    flat = np.full_like(images, 128)
    assert (M.edge_f1(flat, flat) == 1.0).all() and (M.edge_f1(flat, images) == 0.0).all()


def test_calibration_ordering():
    real = make_images()
    candidates = M.reference_candidates(real)
    means = {key: np.mean(M.per_image_metrics(value, real)["ssim"]) for key, value in candidates.items()}
    assert np.mean(M.psnr(candidates["identity"], real)) > 90
    assert means["blur_sigma2"] > means["shuffled_pairs"]
    assert means["blur_sigma2"] > means["dataset_mean_color"]


def test_summarize_per_terrain():
    result = M.summarize({"psnr": np.array([10., 20., 30., 40.]), "ssim": np.array([.1, .2, .3, .4])},
                         ["urban", "urban", "agri", "agri"])
    assert result["all"]["psnr"] == 25.0 and result["urban"]["psnr"] == 15.0
    assert result["agri"]["ssim"] == pytest.approx(0.35)


def test_terrain_from_paths():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "split.txt")
        open(path, "w").write("v_2/urban/s1/a.png\tv_2/urban/s2/a.png\turban\n")
        assert M.terrain_from_paths(path) == ["urban"]


class StubVAE(torch.nn.Module):
    def encode(self, x):
        z = F.avg_pool2d(x, 8)
        return type("O", (), {"latent_dist": type("D", (), {"mean": z})()})()

    def decode(self, z):
        return type("O", (), {"sample": F.interpolate(z, scale_factor=8, mode="nearest")})()


def test_vae_reconstruct_with_stub():
    real = make_images(size=64)
    reconstructed = M.vae_reconstruct(StubVAE(), real, device="cpu", batch=2, scale=1.0)
    assert reconstructed.shape == real.shape and reconstructed.dtype == np.uint8


@pytest.mark.skipif(importlib.util.find_spec("lpips") is not None, reason="lpips installed")
def test_lpips_missing_fails_loudly():
    with pytest.raises(RuntimeError, match="lpips is not installed"):
        M.LPIPSMetric()


def test_eval_script_rejects_float_gen():
    real = make_images()
    with tempfile.TemporaryDirectory() as directory:
        real_path, gen_path = f"{directory}/real.npy", f"{directory}/gen.npy"
        np.save(real_path, real)
        np.save(gen_path, real.astype(np.float32) / 255)
        result = subprocess.run(
            [sys.executable, "scripts/eval_metrics.py", "--real", real_path, "--gen", gen_path,
             "--device", "cpu", "--no-lpips", "--no-fid"],
            capture_output=True, text=True, timeout=240,
        )
        assert result.returncode != 0 and "uint8" in result.stderr
