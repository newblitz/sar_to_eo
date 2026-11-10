!cd /kaggle/working/sar_to_eo

# Confirm dependencies/assets/checkpoint
!python -m sar2eo.preflight --root .

# VAE ceiling and calibration
!python scripts/eval_metrics.py \
  --real /kaggle/input/datasets/prateeking/cache-and-best-check/test_eo.npy\
  --splits splits/val.txt \
  --vae assets/sd-vae-ft-ema \
  --device cuda \
  --out results/ceiling_val.json

!python scripts/eval_metrics.py \
  --real /kaggle/input/datasets/prateeking/cache-and-best-check/test_eo.npy \
  --splits splits/test.txt \
  --vae assets/sd-vae-ft-ema \
  --device cuda \
  --out results/ceiling_test.json

# Validation CFG sweep
for CFG in 1.0 1.5 2.5; do

  echo "======================================"
  echo "Running validation with CFG=$CFG"
  echo "======================================"

  python scripts/generate.py \
    --split val --cfg "$CFG" --steps 50 --seed 0 \
    --checkpoint /kaggle/input/datasets/prateeking/cache-and-best-check/best_ema \
    --batch-size 16 --num-workers 2

  python scripts/eval_metrics.py \
    --real /kaggle/input/datasets/prateeking/cache-and-best-check/val_eo.npy \
    --gen "results/gen_val_cfg${CFG}_steps50.npy" \
    --splits splits/val.txt \
    --device cuda \
    --out "results/metrics_val_cfg${CFG}.json"

done

# command 3
BEST_CFG=1.5   # replace with the best validation CFG

!python scripts/generate.py \
  --split test --cfg "$BEST_CFG" --steps 50 --seed 0 \
  --checkpoint /kaggle/input/datasets/prateeking/cache-and-best-check/best_ema \
  --batch-size 16 --num-workers 2

!python scripts/eval_metrics.py \
  --real /kaggle/input/datasets/prateeking/cache-and-best-check/test_eo.npy \
  --gen "results/gen_test_cfg${BEST_CFG}_steps50.npy" \
  --splits splits/test.txt \
  --device cuda \
  --out results/metrics_test.json

  # command 4

!python scripts/eval_metrics.py \
  --real /kaggle/input/datasets/prateeking/cache-and-best-check/val_eo.npy \
  --splits splits/val.txt \
  --vae assets/sd-vae-ft-ema \
  --device cuda \
  --out results/ceiling_val.json

!python scripts/eval_metrics.py \
  --real /kaggle/input/datasets/prateeking/cache-and-best-check/test_eo.npy \
  --splits splits/test.txt \
  --vae assets/sd-vae-ft-ema \
  --device cuda \
  --out results/ceiling_test.json