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
    from .evaluate import psnr
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.evaluate import psnr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="runs/main/config.json"); ap.add_argument("--split", default="splits/val.txt"); ap.add_argument("--out", default="results/vae_ceiling.json"); a = ap.parse_args()
    cfg = Config.load(a.config); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device); loader = DataLoader(PairedSEN12(a.split), batch_size=1, num_workers=cfg.num_workers)
    scores = []; lpips_metric = None
    try:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        pass
    lpips_scores = []
    for batch in loader:
        eo = batch["eo"].to(device); z, _ = targets.encode(eo, cfg.vae_scale); decoded = targets.decode(z, cfg.vae_scale); scores.append(psnr(decoded, eo))
        if lpips_metric is not None: lpips_scores.append(float(lpips_metric(decoded, eo).mean().item()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"count": len(scores), "psnr": sum(scores) / len(scores) if scores else None, "lpips": sum(lpips_scores) / len(lpips_scores) if lpips_scores else None}, indent=2) + "\n")


if __name__ == "__main__": main()
