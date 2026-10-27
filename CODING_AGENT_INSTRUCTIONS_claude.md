# Coding-Agent Instructions: SAR→EO with SAR-JEPA Features + Representation-Aligned Latent Flow Transformer (Small-Scale Version)

You are a coding agent. Build, train and evaluate the model below end to end. Follow the phases in order. Each phase has **acceptance checks**; do not move on until they pass. Report honestly: never invent numbers, and if something fails or underperforms, say so in the final report.

This is a **small-scale project**: a 16,000-pair terrain-segregated SEN1-2 subset (Kaggle) and a ~130 M-parameter generator that trains on a single GPU.

---

## 0. Goal and deliverables

Translate a Sentinel-1 SAR image into a Sentinel-2 RGB (EO) image.

Model summary:
- A **SAR-JEPA ViT-B/16** encoder turns SAR into tokens.
- Those tokens condition a **latent flow-matching transformer (SiT-B/2, trained from scratch)** that generates EO latents of a **frozen SD-VAE**.
- A **JEPA-style alignment loss** makes an intermediate hidden state of the transformer predict the patch embeddings of the clean EO image from a **frozen DINOv2** encoder. The frozen target encoder avoids cross-modal EMA problems and collapse. This loss also speeds up convergence of a from-scratch generator.
- Loss: `L = L_flow + λ · L_align`, with λ = 0.5. Single training stage with a short warm-up in which the SAR encoder is frozen.

Train **only this one model**. Do not train any baseline, ablation or comparison model.

Deliverables:
1. `sar2eo/` Python package and scripts (layout in section 3).
2. Trained checkpoint (EMA weights) of the main model.
3. `results/metrics.csv`, `results/samples_*.png` (SAR | generated | ground truth grids), `results/loss_curves.png`.
4. `REPORT.md`: what was built, exact configs, metrics, failures and limitations.

---

## 1. Environment

- Python ≥ 3.10, PyTorch ≥ 2.1, **one CUDA GPU with 16–24 GB** is sufficient (SiT-B/2 at 256 px, batch 64, bf16). Use the GPU's native fp16 with a gradient scaler instead of bf16 only if the GPU lacks bf16 support.
- Precision: bf16 autocast, fp32 master weights, fp32 loss computation. TF32 matmuls enabled.
- Install:
```bash
pip install timm diffusers transformers accelerate torchmetrics[image] lpips opencv-python-headless
```
- Clone the **SiT** repository (or the REPA repository, which contains a SiT implementation) for the transformer definition. No pretrained generator checkpoint is used. Check the repository license.
- Fix all seeds, log package versions, and save every config to the run directory.

---

## 2. Pretrained assets (download once, verify, record versions)

| Asset | Source | Used for | Fallback if unavailable |
|---|---|---|---|
| SD-VAE | `stabilityai/sd-vae-ft-ema` (diffusers `AutoencoderKL`) | EO latents (f=8, 4 ch, scale 0.18215) | none needed |
| DINOv2 ViT-B/14 | `torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')` | alignment targets | `timm` `vit_base_patch14_dinov2` |
| SAR-JEPA encoder (ViT-B) | Code: `https://github.com/waterdisappear/SAR-JEPA` (`Pretraining/models_lomar.py`). Weights: **Kaggle Models `liweijie19/sar-jepa`** (or the BaiduYun link in the repo README). Use the main **SAR-JEPA** weights, not the Gaussian-kernel variant. | SAR encoder init | `timm` `vit_base_patch16_224.mae` (ImageNet MAE), clearly labelled as fallback in the report |

The SiT-B/2 generator is trained from scratch (public pretrained SiT checkpoints exist only for the much larger XL/2). Record in `REPORT.md` which SAR encoder init was actually used.

**SAR-JEPA facts to respect (from the repository README):**
- **License: CC BY-NC 4.0.** Use is non-commercial only, with attribution. Record this in `REPORT.md` and cite the SAR-JEPA paper (Li et al., ISPRS J. Photogramm. Remote Sens. 2024).
- It was **pre-trained on SAR target-recognition chips** (MSAR, SAR-Ship, SARSim, SAMPLE), predicting multi-scale SAR gradient features of locally masked patches, and fine-tuned on single-channel amplitude images. It is therefore a local, edge- and scatterer-oriented feature extractor and was not trained on Sentinel-1 land-cover scenes. Do not assume that its features capture scene-level semantics; section 5b adds a cheap check.
- The code base follows **LoMaR** (`timm==0.3.2`) and uses **iRPE relative position encoding**. The checkpoint may therefore not match a stock `timm` ViT (relative-position tables, no absolute positional embedding, or a different image size). Inspect it before loading (section 6.1).

---

## 3. Repository layout

```
sar2eo/
  prepare_data.py     # section 4: pairing, stratified split, quality filter
  dataset.py          # paired dataset with joint augmentation
  sar_encoder.py      # ViT-B/16 wrapper, checkpoint adaptation, LLRD groups
  model.py            # SAR2EO wrapper around the SiT-B/2 backbone
  train.py            # training loop (bf16, EMA, resume)
  sample.py           # Euler sampler with CFG
  evaluate.py         # metrics
  sanity.py           # shape + overfit tests
  configs.py          # all hyperparameters in one dataclass
```

---

## 4. Dataset (Phase 1)

### 4.1 The dataset: Sentinel-1&2 Image Pairs, segregated by terrain (Kaggle)

Source: `https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain` (a subset of SEN1-2 organised into 4 land-cover classes: **agriculture, barren land, grassland, urban**). It contains **4,000 SAR images and 4,000 matching EO images per category, i.e. 16,000 pairs in total**. SEN1-2 patches are 256×256 px, 10 m resolution, Sentinel-1 VV only and Sentinel-2 RGB only. Check the dataset's license on its Kaggle page and record it in `REPORT.md`.

Download with the Kaggle CLI (needs `~/.kaggle/kaggle.json`) or attach the dataset as a notebook input:
```bash
kaggle datasets download -d requiemonk/sentinel12-image-pairs-segregated-by-terrain -p data/ --unzip
```

**Inspect before assuming anything.** List the directory tree, print a few file names per category, and report: the folder layout (which sub-folders hold SAR vs EO), file extension, image size, number of channels and dtype of the SAR and EO files, and the total size on disk. Then:
- Build pairs **within each category** by matching SAR and EO file names (in SEN1-2 the names differ only by `s1`→`s2`). Assert that every SAR file has exactly one EO partner, that the counts are 4,000 per category, and that image sizes match.
- If the SAR images are stored as 3-channel RGB with identical channels, convert them to **single-channel** grayscale (assert the channels are identical first). If they are genuinely 3-channel with different content, stop and report.
- If images are not 256×256, resize both modalities to 256×256 with `cv2.INTER_AREA` and note this in the report.

### 4.2 Splits (per-category stratified)

- Per category: **3,500 train, 150 val, 350 test** → totals **14,000 train, 600 val, 1,400 test**. Fixed seed; save file lists to `splits/{train,val,test}.txt` with the category recorded for each pair.
- **Avoid leakage between splits.** Patches cut from the same Sentinel scene overlap in appearance. If scene identifiers can be parsed from the file names (SEN1-2 names look like `ROIsXXXX_<season>_s1_<scene>_pNNN.png`), assign whole scene groups to a single split and assert zero overlap. If scene ids cannot be parsed, sort the file names within each category and split into **contiguous blocks** (neighbouring file names are likely from the same scene), and state in the report that some residual leakage cannot be excluded.
- **Quality filter:** drop pairs where the EO patch is almost uniform (per-image pixel std < 5 of 255) or the SAR patch is constant (std < 2). Log how many were dropped per category and keep the split sizes balanced by trimming the other splits if needed.
- The terrain category is used **only for stratification and per-terrain reporting**. It is not a model input.

### 4.3 Preprocessing

- Keep **256×256**.
- **SAR in SEN1-2 is already 8-bit scaled intensity**, so scale linearly: `x/127.5 - 1`. Do not apply dB conversion to it. (If the inspection shows raw-power values instead, convert to dB, clip to the training set's 1st–99th percentiles, then scale to [-1, 1].) Record the choice in `configs.py`.
- EO: `x/127.5 - 1` → [-1, 1].
- Augmentation applied **jointly** to the SAR/EO pair: random horizontal and vertical flip, and random 90° rotation (optional). No colour jitter, no independent augmentation of one modality. With a small dataset, augmentation and condition dropout are the main protection against overfitting.
- Loading: decode images in `DataLoader` workers (`num_workers` ≥ 4, `pin_memory=True`, `persistent_workers=True`, `drop_last=True`). Optionally pre-decode once to a uint8 `.npy` memmap (about 4 GB uncompressed for 16,000 pairs) to remove decoding cost.

**Acceptance checks:** 4,000 pairs per category found, split sizes as specified, no scene overlap (or the documented fallback), `results/data_preview.png` with 4 pairs per terrain category, value histograms look sane, measured dataset size printed.

---

## 5. Frozen-model outputs (Phase 2)

The VAE and DINOv2 are frozen. During training compute their outputs on the fly under `torch.no_grad()` and bf16 autocast (this keeps augmentation simple; the cost is small at this scale).

- **VAE target:** `x0 = vae.encode(eo).latent_dist.sample() * 0.18215`, shape (B,4,32,32).
- **DINOv2 target:** resize EO (in [0,1]) to **224×224** (bicubic), normalise with ImageNet mean/std, take `forward_features(x)['x_norm_patchtokens']`, shape (B,256,768) (a 16×16 grid).

**Measure the VAE ceiling first:** encode and decode the 600 val EO images and report PSNR and LPIPS to `results/vae_ceiling.json`. This is the upper bound for any generator built on this VAE. If LPIPS is high, flag it in the report.

**Acceptance checks:** shapes as above, no NaNs, DINOv2 tokens not collapsed (non-trivial std across tokens), VAE ceiling file written.

### 5b. Optional SAR-JEPA feature check (at most ~30 minutes, diagnostic only)

Because SAR-JEPA was pre-trained on target chips, check cheaply whether its frozen features carry scene-level information about the EO image. Freeze the encoder, train a small `Linear` or 2-layer MLP head on the training split to regress the DINOv2 target tokens of the paired EO image from the SAR-JEPA tokens (cosine loss, a few thousand steps), and report the validation cosine similarity in `REPORT.md`. This is a diagnostic of the input features, not a baseline model, and its result does not change the main run. If the cosine similarity is very low, say so as a limitation (the main run still fine-tunes the encoder).

---

## 6. Model specification (Phase 3)

All modules share a 16×16 = 256-token grid.

| Tensor | Shape |
|---|---|
| SAR input | (B, 1, 256, 256) |
| SAR tokens (ViT-B/16) | (B, 256, 768) |
| `cond` (after `cond_proj`) | (B, 256, D), D = 768 |
| global vector `g` | (B, D) |
| EO latent x0 | (B, 4, 32, 32) |
| noisy latent x_t | (B, 4, 32, 32) |
| transformer tokens (patch size 2) | (B, 256, 768) |
| velocity output | (B, 4, 32, 32) |
| aligned hidden state (after block k=4) | (B, 256, 768) via predictor MLP |
| DINOv2 target tokens | (B, 256, 768) |

Backbone: **SiT-B/2**, D = 768, depth 12, heads 12, alignment layer k = 4. (For quick debugging a SiT-S/2 with D = 384, depth 12, heads 6 may be used; set the `cond_proj` width to D and keep `align_proj` output at 768.) Write unit asserts for every row of the shape table in `sanity.py`.

### 6.1 `sar_encoder.py`

**Step 1: inspect the SAR-JEPA checkpoint and code (mandatory, before writing the wrapper).**
- Download the weights (Kaggle Models `liweijie19/sar-jepa`) and clone the repository. Read `Pretraining/models_lomar.py` and the default arguments in `main_pretrain.py`.
- Print the checkpoint's top-level keys, every parameter name and shape for the first block, and the shapes of the patch embedding, any positional embedding and any relative-position (`rpe`) tables. Report: pre-training image size, patch size, embedding dim (expect 768), depth, number of input channels, presence of a CLS token, and whether the encoder uses iRPE.
- Save this report to `results/sar_jepa_checkpoint_report.txt`.

**Step 2: build the encoder, choosing the first route that works.**
1. **Reuse the repository's encoder class** (copy the minimal encoder-only code from `models_lomar.py` into `sar2eo/`, drop the decoder and the gradient-feature targets), load the weights with `strict=False`, and print missing/unexpected keys. This is preferred when the model uses iRPE, because it keeps the pretrained relative-position behaviour exactly.
2. **If the encoder is a plain ViT** (absolute positional embedding, standard blocks), load it into `timm.create_model('vit_base_patch16_224', num_classes=0, global_pool='', class_token=False, img_size=256, in_chans=1, drop_path_rate=0.1)` after stripping prefixes (`module.`, `encoder.`) and dropping `cls_token`. Resample the positional embedding with `timm.layers.resample_abs_pos_embed(pe, (16,16), (old,old), num_prefix_tokens=0)`.
3. **If neither works cleanly,** use the MAE fallback from section 2 and say so in the report.

**Input and output contract (must hold whichever route is used):**
- Input `(B, 1, 256, 256)` in [-1, 1]. If the checkpoint expects a different input size, run it at 256 anyway only if the architecture allows (resampled positional embedding or relative encodings); otherwise resize the SAR to the checkpoint's size for the encoder, then bilinearly resample the output token grid to 16×16. Record whichever is done.
- If the checkpoint patch embedding has a different channel count than 1, sum its weights over channels and repeat/divide by `in_chans`.
- Output `forward(x)` → **(B, 256, 768)** patch tokens (no CLS token). If the encoder's width is not 768, add a `Linear(width, 768)` and state it.
- Keep drop-path small (≤ 0.1). Fine-tune with layer-wise LR decay: `llrd_groups(vit, base_lr, decay=0.75, wd=0.05)` (layer index from `blocks.N` names; no weight decay on biases and norms).

### 6.2 Generator backbone
Use `SiT_B_2(input_size=32, in_channels=4, learn_sigma=False)` from the SiT repository, randomly initialised with the repository's standard init (Xavier for linears, zero for adaLN modulation and the final layer). Verify that these attributes exist: `x_embedder`, `pos_embed`, `t_embedder`, `blocks`, `final_layer`, `unpatchify`. Notes:
- There is no class conditioning. Do not use the label embedder; the adaLN condition is `c = t_emb + g`.
- Time convention: `t ∈ [0,1]`, t = 0 data, t = 1 noise, passed directly to `t_embedder`.
- The output has exactly 4 channels (no learned sigma).

### 6.3 `model.py` (the SAR2EO wrapper)
Implement exactly as follows:
- `cond_proj = LayerNorm(768) → Linear(768,D) → GELU → Linear(D,D)` (**last layer zero-init**).
- `glob_proj = Linear(768, D)` (zero-init) applied to the mean-pooled SAR tokens; the adaLN condition is `c = t_emb + g`.
- `inject = ModuleList([zero-init Linear(D,D)] * depth)`; before every block: `h = h + inject[i](cond)`.
- Input tokens: `h = x_embedder(x_t) + pos_embed + cond`.
- `null_tok (1,256,D)` and `null_glob (1,D)` are learned; with probability 0.1 per sample replace `cond` and `g` with them (condition dropout for classifier-free guidance).
- `align_proj = Linear(D,1024) → SiLU → Linear(1024,1024) → SiLU → Linear(1024,768)` applied to the hidden state after block k (k = 4).
- Methods: `encode_sar(sar) -> (cond, g)`, `denoise(x_t, t, cond, g, drop=None) -> (v, z_or_None)`, `forward(...)`. A `sar_trainable` flag controls whether the SAR encoder receives gradients (use `torch.set_grad_enabled`).
- Return alignment tokens only in training.
- Gradient checkpointing is not needed at this size.

**Acceptance checks (`sanity.py`):** all shapes above; **zero-init property** (at initialisation, replacing `cond` with random tensors does not change the velocity output); forward + backward under bf16 autocast with no NaNs; parameter counts printed (about 130 M for the transformer, about 86 M for the SAR ViT).

---

## 7. Training (Phase 4)

### 7.1 Objective
```python
t   = torch.rand(B)                       # t=0 data, t=1 noise
eps = torch.randn_like(x0)
x_t = (1 - t)[:,None,None,None] * x0 + t[:,None,None,None] * eps
v, z = model(x_t, t, sar, drop_mask)      # drop_mask: Bernoulli(0.1)
L_flow  = F.mse_loss(v.float(), (eps - x0).float())
L_align = -(F.normalize(z.float(), dim=-1) * F.normalize(dino_tokens.float(), dim=-1)).sum(-1).mean()
loss = L_flow + lam * L_align             # lam = 0.5
```

### 7.2 Hyperparameters (`configs.py`)
| Item | Value |
|---|---|
| Batch size | 64 (use gradient accumulation if memory is tight) |
| Optimizer | AdamW (fused), betas (0.9, 0.95), wd 0 on transformer and new modules |
| LR | transformer and new modules **2e-4**; SAR ViT base LR **2e-5** with layer decay 0.75 and wd 0.05 |
| Warm-up | 1000 steps linear |
| SAR encoder frozen | first 3000 steps (`sar_trainable=False`), then unfrozen |
| Grad clip | 1.0 |
| EMA | 0.999; sample and evaluate with EMA weights |
| Precision | bf16 autocast, fp32 master weights, fp32 losses |
| Steps | **60k** to start (about 270 epochs of 14k pairs). Watch validation flow-loss every 2000 steps; stop early or keep the best checkpoint if it starts rising (overfitting). |
| Seeds | fixed (0) |

Speed: `F.scaled_dot_product_attention`, fused AdamW, `torch._foreach_lerp_` for the EMA update. Use `torch.compile` on individual blocks only if it gives a measured speedup.

### 7.3 Robustness requirements
- **Resume:** save `{model, ema, opt, step, rng states}` every 30 minutes and at the end; auto-resume from the latest checkpoint on start. Keep only the last 2 checkpoints plus the best validation one.
- **Logging:** every 100 steps append `step, l_flow, l_align, lr, it/s` to `logs/main.csv`. Every 2000 steps compute validation `L_flow` (fixed noise seed and t grid) with EMA weights and write a 4-sample preview grid.
- **Overfit test first (mandatory):** train on 64 pairs for ~1000 steps. Flow loss must drop clearly and a sampled image must resemble the ground truth. If not, fix the bug before the real run.
- **Stop conditions:** abort and report if the loss becomes NaN, or if `L_align` does not decrease after 5k steps.

---

## 8. Training run (Phase 5)

Train the main model once with the settings above: SAR-JEPA-initialised SAR encoder (or the MAE fallback), SiT-B/2 generator from scratch, DINOv2-B alignment target, λ = 0.5, alignment after block 4. Run the mandatory overfit test first, then the full run. Extend beyond 60k steps only if validation flow-loss is still falling and the previews keep improving.

---

## 9. Sampling and evaluation (Phase 6)

**Sampler** (`sample.py`): Euler ODE from t=1 to t=0, 50 steps (Heun optional), classifier-free guidance scale 1.5 (conditional and unconditional passes batched; encode SAR once and reuse `cond, g` across steps). Decode with `vae.decode(x/0.18215)`.
```python
x = x + (t_next - t) * v      # t decreases from 1 to 0
v = v_uncond + cfg * (v_cond - v_uncond)
```

**Metrics on the 1,400 test pairs** (EMA weights, fixed seeds, one sample per input; plus 4 samples per input on 100 inputs for diversity). Report every metric overall **and per terrain category** (agriculture, barren, grassland, urban):
- **FID** and **KID** against the real test EO. With only 1,400 images (and about 350 per category) FID is biased, so treat it as a rough indicator and report KID alongside it.
- **LPIPS** (AlexNet) to the paired ground truth.
- **PSNR and SSIM** (secondary, since they reward blur).
- **Structure:** Canny edge F1 (OpenCV, fixed thresholds) between generated and real EO on grayscale.
- **Diversity:** mean pairwise LPIPS among the 4 samples of the same SAR input.
- **VAE ceiling** (section 5) reported next to the metrics.
- Sweep CFG scale ∈ {1.0, 1.5, 2.5} on the **val** split only, pick the best scale on val, then evaluate once on test. **Never tune on test.**

Write `results/metrics.csv` and a markdown table into `REPORT.md`. Include paired sample grids, including failure cases.

---

## 10. Optimisation notes

- Run VAE and DINOv2 in bf16 under `no_grad`, in eval mode with `requires_grad_(False)`.
- Use multi-worker data loading from fast local storage, or the pre-decoded `.npy` memmap.
- Encode the SAR once per sample at inference and batch the conditional and unconditional passes.
- If the SAR ViT dominates cost or destabilises early training, freeze it for longer (e.g., 10k steps) before changing the architecture.
- If the generator memorises the training set (validation flow-loss rising while training loss falls), strengthen augmentation, raise condition dropout to 0.15, or stop earlier.

---

## 11. Rules for the agent

1. Inspect before assuming (directory trees, checkpoint keys, module names, tensor shapes). Print them.
2. Make scripts runnable from the command line (`python sar2eo/train.py --run main`), idempotent and resumable.
3. Keep splits leak-free and the test set untouched until final evaluation.
4. Train only the main model described in this file; do not add baseline or ablation runs.
5. If a pretrained asset is missing or does not load cleanly, use the stated fallback and document it; do not silently substitute. Respect the SAR-JEPA CC BY-NC 4.0 license (non-commercial) and cite its paper.
6. In `REPORT.md`, include: configs, final dataset size and counts, wall-clock and GPU hours used, all metrics next to the VAE ceiling, sample grids, what failed, and honest conclusions. Since no baselines are trained, do not claim that any component (SAR-JEPA init, alignment loss) is responsible for the results. With about 14k training pairs, a from-scratch generator will produce plausible but limited-quality output; state this plainly.

---

## 12. Optional extensions (only if explicitly requested)

- Fine-tune the VAE decoder on EO patches if the VAE-ceiling LPIPS is high, then freeze it and rerun the main model.
- Add a cross-attention layer every 4th block if blur or ghosting appears from loosely registered pairs.
- Evaluate the trained model on a different SAR-optical dataset such as QXS-SAROPT or SAR2Opt.
