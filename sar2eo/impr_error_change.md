# SAR-to-EO Pipeline Changes

This document records the changes made while diagnosing and testing the
SAR-to-EO pipeline in the Kaggle SSH/VS Code Remote SSH environment.

Line numbers refer to the current versions of the files at the time this
document was written. They may change after future edits.

## 1. Persistent CUDA library-path fix

### File

`/root/.bashrc` (outside the repository)

### Change

Added the following export after the interactive-shell guard:

```bash
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### Reason

The Kaggle runtime exposed `/dev/nvidia0` and `/dev/nvidia1` and the NVIDIA
driver was loaded, but SSH shells had an empty `LD_LIBRARY_PATH`. As a result,
Python could not discover `/usr/local/nvidia/lib64/libcuda.so.1`, and
`torch.cuda.is_available()` returned `False`.

The export preserves an existing `LD_LIBRARY_PATH` instead of overwriting it.
After the change, a fresh login shell reported:

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 2
torch.cuda.get_device_name(0) = Tesla T4
CUDA tensor allocation and computation = successful
```

## 2. SAR-JEPA NumPy compatibility

### File

[`scripts/kaggle_setup.py`](scripts/kaggle_setup.py), current lines 136-137

### Change

The setup compatibility patch now checks that
`Pretraining/util/pos_embed.py` exists and replaces the removed NumPy 2.x
alias:

```python
if "dtype=np.float)" in text:
    pos_embed.write_text(text.replace("dtype=np.float)", "dtype=float)"))
```

### Reason

The upstream SAR-JEPA source used `np.float`, which raises this error with
modern NumPy:

```text
AttributeError: module 'numpy' has no attribute 'float'
```

Replacing it with the built-in `float` preserves the positional-embedding
calculation and does not alter model architecture, checkpoint keys, or tensor
shapes.

## 3. Idempotent SAR-JEPA timm compatibility patch

### File

[`scripts/kaggle_setup.py`](scripts/kaggle_setup.py), current lines 128-133

### Change

Import replacements are applied only when the replacement has not already
been wrapped:

```python
for old, new in replacements.items():
    if old in text and f"try:\n    {old}" not in text:
        text = text.replace(old, new)
```

### Reason

Kaggle setup may be run repeatedly. The guard prevents nested or malformed
`try/except` blocks from being generated on later runs.

The setup continues to patch the old timm import locations and verifies that
the SAR-JEPA source compiles and that the official factory can instantiate the
required ViT-B/16 encoder.

## 4. Prevent SAR-JEPA from shadowing Python standard-library modules

### File

[`sar2eo/sar_encoder.py`](sar2eo/sar_encoder.py), current line 17

### Change

Changed the SAR-JEPA path registration from inserting at the front of
`sys.path` to appending:

```python
if source not in sys.path:
    sys.path.append(source)
```

### Related setup verification

[`scripts/kaggle_setup.py`](scripts/kaggle_setup.py), current line 156, now
uses:

```python
f"sys.path.append({str(pretraining)!r}); "
```

instead of `sys.path.insert(0, ...)`.

### Reason

The SAR-JEPA `Pretraining` directory contains a file named `profile.py`.
Putting that directory at `sys.path[0]` caused it to shadow Python’s standard
library `profile` module when TorchVision/timm imported `cProfile`. This
produced a failure involving the upstream timm version assertion.

Appending the path still makes `models_lomar` importable while preserving the
normal Python module search order.

## 5. Timestep-shape compatibility for both model paths

### File

[`sar2eo/model.py`](sar2eo/model.py), current lines 39 and 64-65

### Change

The SiT source path is appended instead of inserted at index zero:

```python
if sit_root.exists():
    sys.path.append(str(sit_root))
```

The timestep shape is selected according to the active model implementation:

```python
t_input = t.float().view(b) if self._uses_local_sit else t.float().view(b, 1)
c = self.t_embedder(t_input) + g
```

### Reason

The official SiT `TimestepEmbedder` expects a `(B,)` timestep tensor, while
the local fallback uses `nn.Linear(1, d_model)` and expects `(B, 1)`. A single
unconditional reshape caused one of the two paths to fail with a matrix
dimension error.

This change only adapts the input shape and does not change model layers,
weights, or architecture.

## 6. One-sample GPU end-to-end test

### File

[`sar2eo/sample_test.py`](sar2eo/sample_test.py), new file, current lines 1-110

### Behavior

The test:

1. Requires CUDA and intentionally does not fall back to CPU.
2. Validates the requested CUDA device index.
3. Loads exactly one row from the selected split.
4. Loads the official SAR-JEPA checkpoint.
5. Loads the frozen SD-VAE and DINOv2 targets.
6. Runs the SAR-to-EO model forward pass.
7. Computes flow-matching and representation-alignment losses.
8. Checks that the loss is finite.
9. Runs backward propagation.
10. Checks that trainable gradients exist and are finite.
11. Executes one AdamW optimizer step.
12. Prints a structured `status: PASS` result.

Run it with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sar2eo.sample_test \
  --split splits/train.txt \
  --index 0 \
  --device 0
```

The test passed on a Tesla T4 with these shapes:

```text
sar:      (1, 1, 256, 256)
eo:       (1, 3, 256, 256)
latent:   (1, 4, 32, 32)
dino:     (1, 256, 768)
velocity: (1, 4, 32, 32)
aligned:  (1, 256, 768)
```

## 7. Training alignment-loss dimension bug

### File

[`sar2eo/train.py`](sar2eo/train.py), current lines 282-284

### Change

Changed:

```python
F.normalize(z.float(), -1) * F.normalize(dino.float(), -1)
```

to:

```python
F.normalize(z.float(), dim=-1) * F.normalize(dino.float(), dim=-1)
```

### Reason

In PyTorch, the second positional argument to `F.normalize` is `p`, the norm
order. It is not the dimension argument. Therefore, `-1` was interpreted as
`p=-1`, producing an invalid alignment loss around `-225186752`.

Using the explicit `dim=-1` argument correctly normalizes each feature vector
over its embedding dimension. After the fix, the 100-step training run
reported:

```text
step=100 flow=1.43612 align=-0.63335 lr=2.000e-05
```

The alignment value is now in the expected bounded cosine-similarity range.

## 8. Documentation update

### File

[`KAGGLE_SETUP.md`](KAGGLE_SETUP.md), current lines 72 and 87-111

### Change

Documented:

- The NumPy/timm compatibility patches performed by setup.
- The new one-sample GPU test.
- The requirement to obtain `status: PASS` before distributed training.
- Renumbered the later setup sections.

## 9. Validation performed

The following checks completed successfully:

```text
Python compilation and git diff --check
CUDA visibility and actual CUDA tensor computation
SAR-JEPA factory import
One-sample GPU end-to-end test
Debug model sanity test
Two-GPU distributed smoke test
One-step GPU training
100-step GPU overfit training
Checkpoint resume from step 100
One-sample VAE ceiling evaluation
Ten-sample model evaluation
```

The full 1,400-sample evaluation was started but stopped after approximately
19 minutes without producing a result. The bounded ten-sample evaluation
completed successfully. The full evaluation is therefore not claimed as
completed.

Non-fatal warnings remain from upstream or environment components:

- Deprecated timm import paths.
- Optional SAR-JEPA `rpe_ops` extension not being built.
- DataLoader worker-count advisories.
- DDP gradient-stride performance warnings.

These warnings did not prevent the tested forward, backward, optimizer,
checkpoint, resume, or bounded evaluation paths from completing.
