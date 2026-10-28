# SAR→EO implementation report

Date: 2026-10-28

## Executive summary

The repository was updated for the small-scale SAR→EO specification and the uploaded `v_2` dataset was inspected and split. The local SD-VAE, DINOv2, SiT source, and SAR-JEPA checkpoint were staged and validated. Dataset preparation, preflight, frozen-asset loading, SAR-JEPA loading, EMA architecture matching, and CPU sanity checks passed. Full model training and final evaluation have not run because this development host has no CUDA GPU.

No trained-model metrics or image-quality claims are made.

## Work completed

### Repository and configuration

- Implemented the `sar2eo/` package and command-line scripts.
- Updated defaults to SiT-B/2 dimensions: D=768, depth=12, heads=12, alignment layer 4.
- Set the requested training configuration: batch target 64, LR 2e-4 for new/generator modules, SAR LR 2e-5, 1,000-step warmup, 3,000-step SAR freeze, 60,000 steps, EMA 0.999, alignment weight 0.5.
- Added configuration serialization and CPU-compatible debug fallback behavior.
- Added Kaggle setup automation in `scripts/kaggle_setup.py` and `KAGGLE_SETUP.md`.
- Added two-GPU DDP training through `torchrun --nproc_per_node=2`, with distributed sampling, rank-safe checkpoints, and gradient accumulation preserving global batch size 64.
- Added a safe TorchVision compatibility helper that avoids duplicate `nms`/`qnms` registration.

Relevant files: [configs.py](./sar2eo/configs.py), [model.py](./sar2eo/model.py), [sar_encoder.py](./sar2eo/sar_encoder.py), [requirements.txt](./requirements.txt).

### Dataset inspection and preparation

The uploaded [v_2/](./v_2/) directory contains:

| Terrain folder | SAR files | EO files | Pair count |
|---|---:|---:|---:|
| `agri` | 4,000 | 4,000 | 4,000 |
| `barrenland` | 4,000 | 4,000 | 4,000 |
| `grassland` | 4,000 | 4,000 | 4,000 |
| `urban` | 4,000 | 4,000 | 4,000 |

Observed properties:

- 32,000 PNG files total.
- Images are 256×256, 8-bit RGB PNGs.
- SAR images have identical RGB channels and are safely converted to one grayscale channel.
- Total selected pair size is 2,730,627,140 bytes, or 2.543 GiB.
- Quality filtering dropped no pairs.

Generated artifacts:

- [splits/train.txt](./splits/train.txt): 14,000 rows.
- [splits/val.txt](./splits/val.txt): 600 rows.
- [splits/test.txt](./splits/test.txt): 1,400 rows.
- [splits/selection.json](./splits/selection.json): counts, size, and split method.
- [results/data_preview.png](./results/data_preview.png): 16 terrain-labelled SAR|EO examples.

The split is balanced at 3,500 train, 150 validation, and 350 test examples per terrain.

### Validation performed

- Dataset pair discovery and image-size checks passed.
- SAR identical-channel check passed.
- Split row counts passed.
- Dataset loader returned SAR `(1,256,256)` and EO `(3,256,256)` tensors in `[-1,1]`.
- All Python files passed `py_compile` using `/opt/anaconda3/bin/python`.
- CPU debug sanity check passed with velocity `(2,4,32,32)` and alignment tokens `(2,256,768)`.
- Local SD-VAE loaded and produced `(B,4,32,32)` latents; local DINOv2 loaded and produced `(B,256,768)` tokens.
- Supplied SAR-JEPA checkpoint loaded with no missing compatible weights. Its iRPE lookup-table weights are reported but not applied by the compact encoder.
- EMA architecture was verified by copying the already SAR-JEPA-adapted base model.
- All command-line entry points passed help/compile checks.
- [results/data_preview.png](./results/data_preview.png) was visually inspected.

## Assumptions

1. `agri` maps to agriculture and `barrenland` maps to barren.
2. `s1/` and `s2/` identify SAR and EO respectively.
3. SAR PNGs are already 8-bit scaled intensity, so preprocessing uses `x/127.5 - 1` without dB conversion.
4. The filename pattern provides a candidate scene key.
5. Terrain is metadata only and is not passed to the model.
6. The supplied SAR-JEPA checkpoint is the intended SAR initialization. Its compact compatible encoder ignores the checkpoint's iRPE lookup tables, which is recorded as a limitation.
7. CPU debug execution is sufficient for shape/syntax checks, but not for performance or quality conclusions.

## Problems encountered

### Runtime and hardware

- The SD-VAE and DINOv2 weights are now present under `assets/` and load with `local_files_only=True`.
- The SiT source is cloned under `assets/SiT` and integrated as the generator backbone.
- The SAR-JEPA source is not currently retained in this local checkout, but `scripts/kaggle_setup.py` clones the official repository into `assets/SAR-JEPA` on Kaggle before training.
- The SAR-JEPA checkpoint is present under `assets/sar_jepa/checkpoint-200.pth`.
- The development host has no CUDA GPU, so the two-GPU Kaggle path could not be executed locally.

### Unfinished outputs

Not generated:

- `results/vae_ceiling.json`
- EMA training checkpoint
- 1,000-step overfit result
- 60,000-step model run
- Actual metrics in `results/metrics.csv`
- `results/samples_*.png`
- `results/loss_curves.png`
- Per-terrain FID, KID, LPIPS, PSNR, SSIM, edge F1, and diversity metrics
- SAR-JEPA feature diagnostic

## Contradictions and specification risks

### Whole-scene split versus exact counts

The instructions require both exact per-terrain counts of 3,500/150/350 and whole scene groups assigned to one split. The uploaded data contains only a few very large scene groups. For example, agriculture includes groups with approximately 1,370, 886, 800, 527, 288, and 129 patches. Those sizes cannot produce exact 3,500/150/350 totals while keeping every scene intact.

The implementation uses sorted contiguous filename blocks as a fallback and records `scene_split_fallback: true` for every terrain in [selection.json](./splits/selection.json). Counts are exact, but residual scene leakage risk remains.

### External SiT repository versus local implementation

The external SiT repository was cloned under `assets/SiT`, and the local `SiT_B_2` class is integrated and used when the default dimensions are selected. The Kaggle setup script reclones it when the GitHub checkout does not contain the ignored source directory.

### Metrics versus evaluation infrastructure

Evaluation hooks exist, but actual FID/KID, LPIPS, SSIM, CFG sweep, per-terrain metrics, diversity samples, and failure-case grids have not been run. The evaluation code is infrastructure, not evidence of final quality.

## License and provenance

The instructions identify SAR-JEPA weights as CC BY-NC 4.0 and require non-commercial use with attribution. The supplied SAR-JEPA checkpoint is used and must retain that attribution/non-commercial restriction. The SiT source is MIT licensed. The uploaded dataset’s Kaggle license was not independently retrieved and must be checked before redistribution or commercial use.

## Conclusion

Dataset preparation and local CPU-side validation are complete and reproducible for the uploaded files. The Kaggle setup and two-GPU training path are implemented, but this is not yet a trained or evaluated SAR→EO result. The main unresolved dataset risk is the unavoidable compromise between exact balanced counts and whole-scene leakage prevention in this dataset layout.
