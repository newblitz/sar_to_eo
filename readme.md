# SAR-to-EO Image Translation

SAR-to-EO is a conditional generative model that translates single-channel
Synthetic Aperture Radar (SAR) images into corresponding RGB optical
images. The project is designed for paired SAR and Earth-observation imagery
from the SEN1-2 dataset.

![Model architecture](model_archietecture/sar2eo_paper_figure_architecture.png)

## Architecture

The model consists of four main components:

1. **SAR encoder** - A SAR-JEPA/LoMaR ViT-B/16 encoder extracts spatial
   features from the input SAR image.
2. **Conditioning projections** - SAR token features and global features are
   projected into the latent flow-transformer representation.
3. **Latent flow transformer** - A SiT-B/2 backbone operates on SD-VAE latent
   representations and predicts the flow velocity needed to generate the EO
   image.
4. **Frozen target encoders** - SD-VAE provides image latents, while DINOv2
   provides semantic image representations for training targets.

Training combines flow-matching loss with a DINOv2 representation-alignment
loss. The SAR-JEPA encoder and the target VAE/DINOv2 models are initialized
from pretrained checkpoints.

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/newblitz/sar_to_eo
cd sar_to_eo
pip install -r requirements.txt
```

For the complete CUDA/Kaggle setup, attach the SEN1-2 dataset and SAR-JEPA
checkpoint, then run:

```bash
pip install -r requirements-kaggle.txt
python scripts/kaggle_setup.py \
  --repo-root /kaggle/working/sar_to_eo \
  --sar-jepa-input /kaggle/input/<checkpoint-dataset>/checkpoint-200.pth
```

Prepare the dataset and generate the train, validation, and test splits:

```bash
python -m sar2eo.prepare_data --root v_2 --seed 0
python -m sar2eo.preflight --root .
```

## Training

Run the recommended one-sample and distributed smoke tests before training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sar2eo.sample_test \
  --split splits/train.txt --index 0 --device 0

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.smoke_test --split splits/train.txt
```

Start training with:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m sar2eo.train --run main --steps 60000
```

Training automatically resumes from the latest checkpoint in
`runs/main/checkpoint_last.pt`.

## Evaluation

Evaluate a trained checkpoint with:

```bash
python -m sar2eo.evaluate \
  --checkpoint runs/main/checkpoint_last.pt \
  --config runs/main/config.json \
  --split splits/test.txt \
  --out results/metrics.csv
```

## Results

The current reported results are:

| Metric | Score |
|---|---:|
| PSNR | 18.4 dB |
| LPIPS | 0.417 |
