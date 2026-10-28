import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

try:
    from .configs import Config
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sar_encoder import load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sar_encoder import load_sar_checkpoint


def load_sar(path, device):
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return torch.from_numpy(x).div(127.5).sub(1).unsqueeze(0).unsqueeze(0).to(device)


@torch.no_grad()
def sample(model, sar, steps=50, cfg_scale=1.5):
    cond, g = model.encode_sar(sar); x = torch.randn(sar.shape[0], 4, 32, 32, device=sar.device)
    times = torch.linspace(1, 0, steps + 1, device=sar.device)
    model.eval()
    for t, tn in zip(times[:-1], times[1:]):
        tb = torch.full((sar.shape[0],), t, device=sar.device); un, _ = model.denoise(x, tb, cond, g, torch.ones(sar.shape[0], device=sar.device, dtype=torch.bool)); co, _ = model.denoise(x, tb, cond, g, torch.zeros(sar.shape[0], device=sar.device, dtype=torch.bool))
        x = x + (tn - t) * (un + cfg_scale * (co - un))
    return x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--sar", required=True); ap.add_argument("--output", default="results/sample.png"); ap.add_argument("--config", default="runs/main/config.json"); ap.add_argument("--cfg", type=float, default=1.5); a = ap.parse_args()
    cfg = Config.load(a.config); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim).to(device)
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    state = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("ema", state.get("model", state)))
    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device); z = sample(model, load_sar(a.sar, device), cfg_scale=a.cfg); image = ((targets.decode(z, cfg.vae_scale)[0] + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy(); Path(a.output).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image).save(a.output)


if __name__ == "__main__": main()
