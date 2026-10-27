# Kaggle setup and run commands

The GitHub repository intentionally excludes the dataset and model weights. Attach
the SEN1-2 dataset and the SAR-JEPA checkpoint as Kaggle inputs, then run these
cells from a Kaggle notebook.

## 1. Clone the repository

Replace YOUR_GITHUB_REPO_URL with the repository URL:

~~~bash
%cd /kaggle/working
!git clone YOUR_GITHUB_REPO_URL sar-to-eo
%cd /kaggle/working/sar-to-eo
~~~

## 2. Install dependencies

~~~bash
!pip install -q -U \
  "torch>=2.1" timm diffusers transformers huggingface_hub accelerate \
  "torchmetrics[image]" lpips opencv-python-headless Pillow numpy
~~~

## 3. Prepare the exact local structure

The setup script searches /kaggle/input for folders named agri, barrenland,
grassland, and urban, then creates:

~~~text
v_2/                              -> Kaggle dataset input
assets/SiT/                       -> cloned SiT source
assets/SAR-JEPA/                 -> cloned SAR-JEPA source
assets/sd-vae-ft-ema/            -> downloaded SD-VAE
assets/dinov2-base/              -> downloaded DINOv2
assets/sar_jepa/checkpoint-200.pth -> attached SAR-JEPA checkpoint
~~~

~~~bash
!python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar-to-eo
~~~

If the dataset is nested differently, pass its exact directory:

~~~bash
!python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar-to-eo \
  --dataset-input /kaggle/input/<dataset-directory>
~~~

If the checkpoint has a nonstandard filename or location:

~~~bash
!python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar-to-eo \
  --sar-jepa-input /kaggle/input/<checkpoint-dataset>/checkpoint-200.pth
~~~

The script fails instead of silently substituting a different SAR checkpoint.

## 4. Prepare data and run preflight

~~~bash
%cd /kaggle/working/sar-to-eo
!python -m sar2eo.prepare_data --root . --seed 0
!python -m sar2eo.preflight --root .
!python -m sar2eo.sanity
~~~

Preflight must show CUDA available, the Kaggle GPU name, the local VAE and
DINOv2 directories, and the SAR-JEPA checkpoint.

## 5. Measure the frozen VAE ceiling

~~~bash
!python -m sar2eo.vae_ceiling \
  --config /kaggle/working/sar-to-eo/runs/main/config.json \
  --split /kaggle/working/sar-to-eo/splits/val.txt \
  --out /kaggle/working/sar-to-eo/results/vae_ceiling.json
~~~

Check the available options if needed:

~~~bash
!python -m sar2eo.vae_ceiling --help
~~~

## 6. Run the required overfit/debug check

~~~bash
!python -m sar2eo.train --run overfit --overfit --steps 1000
~~~

Inspect runs/overfit/logs.csv and confirm that the loss decreases before
starting the full run.

## 7. Run the main training job

~~~bash
!python -m sar2eo.train --run main --steps 60000
~~~

The default configuration uses 256px images, latent size 32, SiT-B/2
dimensions, 1,000-step warmup, 3,000-step SAR freeze, EMA decay 0.999, and
alignment weight 0.5.

To resume after a Kaggle session interruption, rerun the setup cells and use:

~~~bash
!python -m sar2eo.train --run main --steps 60000
~~~

## 8. Evaluate and save results

~~~bash
!python -m sar2eo.evaluate \
  --checkpoint runs/main/checkpoint_last.pt \
  --config runs/main/config.json \
  --split splits/test.txt \
  --out results/metrics.csv
~~~

Generate one sample:

~~~bash
!python -m sar2eo.sample \
  --checkpoint runs/main/checkpoint_last.pt \
  --config runs/main/config.json \
  --sar /kaggle/working/sar-to-eo/v_2/agri/s1/<sar-file>.png \
  --output results/sample_agri.png \
  --cfg 1.5
~~~

## 9. Persist outputs

Kaggle sessions are temporary. Save runs/, results/, and REPORT.md as notebook
output or copy them to a Kaggle Dataset after training. Never commit v_2/,
downloaded weights, or checkpoints to GitHub.
