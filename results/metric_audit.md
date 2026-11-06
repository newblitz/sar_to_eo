# Metric audit

The evaluation path now uses [`sar2eo/metrics_fixed.py`](../sar2eo/metrics_fixed.py)
for every image metric. Both real and generated values are uint8
`(N,H,W,3)` arrays; model tensors are converted once with `m11_to_u8`.

| Metric | Input/range | Dtype | `data_range` | Aggregation | Channels | Clamp | Matches required convention |
|---|---|---|---|---|---|---|---|
| PSNR | uint8, converted to [0,1] | uint8 | 1 | per-image, then mean | RGB | no metric clamp | Yes |
| SSIM | uint8, converted to [0,1] | uint8 | 1 | per-image, then mean | RGB, per-channel mean | no metric clamp | Yes |
| Edge F1 (2 px) | uint8 RGB | uint8 | n/a | per-image, then mean | grayscale Canny conversion | no | Yes |
| Edge F1 (exact) | uint8 RGB | uint8 | n/a | per-image, then mean | grayscale Canny conversion | no | Yes |
| LPIPS | uint8 converted to [-1,1] | uint8 | n/a | per-image, then mean | RGB | conversion is bounded | Yes |
| FID | uint8 BCHW for torchmetrics | uint8 | n/a | pooled feature distribution | RGB | no | Yes |
| KID | uint8 BCHW for torchmetrics | uint8 | n/a | pooled feature distribution | RGB | no | Yes |

## Legacy deviations

The old [`sar2eo/evaluate.py`](../sar2eo/evaluate.py) used a tensor PSNR helper
and an absolute-error SSIM proxy, plus exact-pixel Canny matching. The old
[`sar2eo/vae_ceiling.py`](../sar2eo/vae_ceiling.py) averaged a tensor PSNR and
silently skipped LPIPS when its package was unavailable. These calls are now
routed through `metrics_fixed`.

The old PSNR path operated on model tensors in `[-1,1]` while using
`data_range=1` in its `[0,1]` formula. That is the classic range error and
lowers PSNR by exactly 6.0206 dB. The corrected path converts to uint8 first
and uses `data_range=1` only after conversion to `[0,1]`.

Missing LPIPS, torchmetrics, or torch-fidelity dependencies now raise an
explicit error. The `--no-lpips` and `--no-fid` options in `eval_metrics.py`
are reserved for dependency-free tests and are not used for reported runs.
