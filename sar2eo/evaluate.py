import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sample import sample
    from .sar_encoder import load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sample import sample
    from sar2eo.sar_encoder import load_sar_checkpoint


def psnr(a, b): return float((-10 * torch.log10(F.mse_loss(a, b).clamp_min(1e-8))).item())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--config", default="runs/main/config.json"); ap.add_argument("--split", default="splits/test.txt"); ap.add_argument("--out", default="results/metrics.csv"); a = ap.parse_args()
    cfg = Config.load(a.config); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim)
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    model = model.to(device)
    state = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema", state.get("model", state)))
    model.eval()
    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device); loader = DataLoader(PairedSEN12(a.split, False), batch_size=1, num_workers=cfg.num_workers, pin_memory=True)
    vals = {"psnr": [], "lpips": [], "ssim": [], "edge_f1": []}
    try: import cv2
    except ImportError: cv2 = None
    for batch in loader:
        sar, gt = batch["sar"].to(device), batch["eo"].to(device); pred = targets.decode(sample(model, sar), cfg.vae_scale)
        vals["psnr"].append(psnr(pred, gt)); vals["ssim"].append(float((1 - (pred - gt).abs().mean()).clamp(0, 1).item()))
        if cv2 is not None:
            pa = cv2.Canny(((pred[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8), 50, 150) > 0; ga = cv2.Canny(((gt[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8), 50, 150) > 0; vals["edge_f1"].append(float(2 * (pa & ga).sum() / (pa.sum() + ga.sum() + 1e-8)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); [w.writerow([k, float(np.mean(v)) if v else "unavailable"]) for k, v in vals.items()]
    print({k: float(np.mean(v)) if v else None for k, v in vals.items()})


if __name__ == "__main__": main()
