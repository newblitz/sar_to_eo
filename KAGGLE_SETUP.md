# Kaggle setup and run commands

The GitHub repository intentionally excludes the dataset and model weights. Attach
the SEN1-2 dataset and the SAR-JEPA checkpoint as Kaggle inputs, then run these
cells from a Kaggle notebook.

## 1. Clone the repository

Replace YOUR_GITHUB_REPO_URL with the repository URL:

~~~bash
%cd /kaggle/working
!git clone https://github.com/newblitz/sar_to_eo sar_to_eo
%cd /kaggle/working/sar_to_eo
~~~

## 2. Install dependencies

Kaggle already supplies the CUDA-enabled PyTorch build. Do not reinstall
PyTorch, and remove Kaggle's preinstalled TorchAudio because this project has
no audio input and the preinstalled TorchAudio may target a different CUDA
version.

~~~bash
!python -m pip uninstall -y torchaudio
!python -m pip install -q -U -r requirements-kaggle.txt
!python -c "import importlib.util, torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() >= 2; assert importlib.util.find_spec('torchaudio') is None; from diffusers import AutoencoderKL; from transformers import Dinov2Model; print('diffusers/transformers imports: OK')"
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
  --repo-root /kaggle/working/sar_to_eo \
  --sar-jepa-input /kaggle/input/datasets/prateeking/sar-jepa-input/checkpoint-200.pth
~~~

If the dataset is nested differently, pass its exact directory:

~~~bash
!python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar_to_eo \
  --dataset-input /kaggle/input/<dataset-directory>
~~~

If the checkpoint has a nonstandard filename or location:

~~~bash
!python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar_to_eo \
  --sar-jepa-input /kaggle/input/<checkpoint-dataset>/checkpoint-200.pth
~~~

The script fails instead of silently substituting a different SAR checkpoint.

## 4. Prepare data and run preflight

~~~bash
%cd /kaggle/working/sar_to_eo
!python -m sar2eo.prepare_data --root v_2 --seed 0
!python -m sar2eo.preflight --root .
!python -m sar2eo.sanity
~~~

Preflight must show CUDA available, the Kaggle GPU name, the local VAE and
DINOv2 directories, and the SAR-JEPA checkpoint.

## 5. Run one two-GPU smoke test

This performs one real distributed forward and backward pass through the
dataset, official SAR-JEPA/iRPE encoder, frozen SD-VAE, frozen DINOv2,
SiT-B/2 path, and both training losses. It does not create checkpoints or
enter the 60,000-step training loop. Require a `status: PASS` result before
starting a training run.

~~~bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.smoke_test \
  --split splits/train.txt
~~~

## 6. Measure the frozen VAE ceiling

~~~bash
!CUDA_VISIBLE_DEVICES=0 python -m sar2eo.vae_ceiling \
  --split /kaggle/working/sar_to_eo/splits/val.txt \
  --out /kaggle/working/sar_to_eo/results/vae_ceiling.json
~~~

Check the available options if needed:

~~~bash
!python -m sar2eo.vae_ceiling --help
~~~

## 7. Run the required overfit/debug check

~~~bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.train --run overfit --overfit --steps 1000
~~~

Inspect runs/overfit/logs.csv and confirm that the loss decreases before
starting the full run.

## 8. Run the main training job

~~~bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.train --run main --steps 60000
~~~

The default configuration uses 256px images, latent size 32, SiT-B/2
dimensions, 1,000-step warmup, 3,000-step SAR freeze, EMA decay 0.999, and
alignment weight 0.5.

To resume after a Kaggle session interruption, rerun the setup cells and use:

~~~bash
!CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.train --run main --steps 60000
~~~

## 9. Evaluate and save results

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
  --sar /kaggle/working/sar_to_eo/v_2/agri/s1/<sar-file>.png \
  --output results/sample_agri.png \
  --cfg 1.5
~~~

## 10. Persist outputs

Kaggle sessions are temporary. Save runs/, results/, and REPORT.md as notebook
output or copy them to a Kaggle Dataset after training. Never commit v_2/,
downloaded weights, or checkpoints to GitHub.
