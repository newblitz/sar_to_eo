"""Recompute calibration, VAE-ceiling, and generated-image metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from sar2eo import metrics_fixed as M


def table(name, per_image, terrain):
    summary = M.summarize(per_image, terrain)
    print(f"\n[{name}]")
    for category, values in summary.items():
        print(f"  {category:12s} " + "  ".join(f"{key}={value:.4f}" for key, value in values.items()))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="uint8 (N,H,W,3) real EO array")
    parser.add_argument("--gen", help="uint8 generated EO array in split order")
    parser.add_argument("--splits", help="split file for per-terrain metrics")
    parser.add_argument("--vae", help="local VAE directory; computes the VAE ceiling")
    parser.add_argument("--out", default="results/metrics_fixed.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-lpips", action="store_true", help="only for dependency-free smoke tests")
    parser.add_argument("--no-fid", action="store_true", help="only for dependency-free smoke tests")
    parser.add_argument("--max-n", type=int)
    args = parser.parse_args()

    real = np.asarray(np.load(args.real, mmap_mode="r")[: args.max_n])
    if real.dtype != np.uint8 or real.ndim != 4 or real.shape[-1] != 3:
        raise ValueError(f"real images must be uint8 (N,H,W,3), got {real.dtype} {real.shape}")
    terrain = M.terrain_from_paths(args.splits)[: len(real)] if args.splits else None
    if terrain is not None and len(terrain) != len(real):
        raise ValueError(f"split rows ({len(terrain)}) != real images ({len(real)})")

    lpips_metric = None if args.no_lpips else M.LPIPSMetric(args.device)
    result = {"n": len(real), "real_color": M.color_stats(real), "calibration": {}}
    for name, candidate in M.reference_candidates(real).items():
        result["calibration"][name] = table(
            f"reference: {name}", M.per_image_metrics(candidate, real, lpips_metric), terrain
        )

    if args.vae:
        from diffusers import AutoencoderKL

        reconstruction = M.vae_reconstruct(
            AutoencoderKL.from_pretrained(args.vae, local_files_only=True),
            real,
            args.device,
        )
        result["vae_ceiling"] = table(
            "VAE ceiling", M.per_image_metrics(reconstruction, real, lpips_metric), terrain
        )
        result["recon_color"] = M.color_stats(reconstruction)

    if args.gen:
        generated_file = np.load(args.gen, mmap_mode="r")
        if len(generated_file) != len(real):
            raise ValueError(f"generated rows ({len(generated_file)}) != real rows ({len(real)})")
        generated = np.asarray(generated_file)
        if generated.dtype != np.uint8 or generated.shape != real.shape:
            raise ValueError(
                f"generated images must be uint8 with shape {real.shape}, got "
                f"{generated.dtype} {generated.shape}"
            )
        result["generated"] = table(
            "generated vs real", M.per_image_metrics(generated, real, lpips_metric), terrain
        )
        result["gen_color"] = M.color_stats(generated)
        if not args.no_fid:
            fid, kid_mean, kid_std = M.fid_kid(real, generated, args.device)
            result.update(fid=fid, kid_mean=kid_mean, kid_std=kid_std)
            print(f"\nFID={fid:.2f}  KID={kid_mean:.4f}+-{kid_std:.4f}")
        print("\ncolour stats real:", result["real_color"], "generated:", result["gen_color"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as output:
        json.dump(result, output, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
