"""Measure the frozen VAE reconstruction ceiling on the supplied validation split."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from . import metrics_fixed as M
    from .progress import Progress
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo import metrics_fixed as M
    from sar2eo.progress import Progress


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="runs/main/config.json"); ap.add_argument("--split", default="splits/val.txt"); ap.add_argument("--out", default="results/vae_ceiling.json"); a = ap.parse_args()
    cfg_path = Path(a.config)
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config()
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}; using default Config for frozen-asset evaluation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device); loader = DataLoader(PairedSEN12(a.split), batch_size=1, num_workers=cfg.num_workers)
    lpips_metric = M.LPIPSMetric(str(device))
    reconstructions, real_images = [], []
    progress = Progress(len(loader), "VAE ceiling")
    for batch in loader:
        eo = batch["eo"].to(device)
        with torch.no_grad():
            z = targets.vae.encode(eo).latent_dist.mean * cfg.vae_scale
            decoded = targets.decode(z, cfg.vae_scale)
        reconstructions.append(M.m11_to_u8(decoded))
        real_images.append(M.m11_to_u8(eo))
        progress.update()
    reconstructed = np.concatenate(reconstructions)
    originals = np.concatenate(real_images)
    values = M.per_image_metrics(reconstructed, originals, lpips_metric)
    summary = M.summarize(values)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"count": len(originals), **summary["all"]}, indent=2) + "\n")


if __name__ == "__main__": main()
