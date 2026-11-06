import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sample import sample
    from .sar_encoder import load_sar_checkpoint
    from . import metrics_fixed as M
    from .progress import Progress
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sample import sample
    from sar2eo.sar_encoder import load_sar_checkpoint
    from sar2eo import metrics_fixed as M
    from sar2eo.progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="splits/test.txt")
    ap.add_argument("--out", default="results/metrics.csv")
    a = ap.parse_args()
    checkpoint = Path(a.checkpoint)
    config_path = Path(a.config) if a.config else checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No configuration was found for checkpoint {checkpoint}. "
            f"Pass --config explicitly; looked for {config_path}."
        )
    cfg = Config.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim)
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    model = model.to(device)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema", state.get("model", state)))
    model.eval()
    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device)
    loader = DataLoader(
        PairedSEN12(a.split, False, cfg.image_size),
        batch_size=1,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    generated, real = [], []
    lpips_metric = M.LPIPSMetric(str(device))
    progress = Progress(len(loader), "evaluation")
    for batch in loader:
        sar, gt = batch["sar"].to(device), batch["eo"].to(device); pred = targets.decode(sample(model, sar), cfg.vae_scale)
        generated.append(M.m11_to_u8(pred))
        real.append(M.m11_to_u8(gt))
        progress.update()
    metrics = M.summarize(M.per_image_metrics(np.concatenate(generated), np.concatenate(real), lpips_metric))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); [w.writerow([k, v]) for k, v in metrics["all"].items()]
    print(metrics["all"])


if __name__ == "__main__": main()
