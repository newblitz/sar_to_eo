import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sar_encoder import llrd_groups, load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sar_encoder import llrd_groups, load_sar_checkpoint


def update_ema(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()): e.lerp_(p.detach(), 1 - decay)


def save_checkpoint(path, model, ema, opt, step):
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "rng": {"torch": torch.get_rng_state(), "python": random.getstate()}}, path)


def latest_checkpoint(run: Path):
    paths = sorted(run.glob("checkpoint_*.pt")) + ([run / "checkpoint_last.pt"] if (run / "checkpoint_last.pt").exists() else [])
    return paths[-1] if paths else None


def plot_loss_curves(logs, output):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        data = np.loadtxt(logs, delimiter=",", skiprows=1)
        if data.ndim == 1: data = data[None, :]
        plt.plot(data[:, 0], data[:, 1], label="flow"); plt.plot(data[:, 0], data[:, 2], label="alignment"); plt.xlabel("step"); plt.legend(); plt.tight_layout(); plt.savefig(output); plt.close()
    except Exception as exc:
        print("loss curve not written:", exc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="main"); ap.add_argument("--config"); ap.add_argument("--steps", type=int)
    ap.add_argument("--overfit", action="store_true"); a = ap.parse_args()
    cfg = Config.load(a.config) if a.config else Config(output_root=f"runs/{a.run}")
    if a.overfit: cfg.total_steps = 1000
    if a.steps: cfg.total_steps = a.steps
    random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.backends.cuda.matmul.allow_tf32 = True
    run = Path(cfg.output_root); run.mkdir(parents=True, exist_ok=True); cfg.save(run / "config.json")
    split = Path(cfg.split_root) / "train.txt"
    if not split.exists(): raise FileNotFoundError(f"Missing {split}; provide the dataset and run prepare_data.py.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedSEN12(split, augment=True, image_size=cfg.image_size)
    if a.overfit: dataset = Subset(dataset, list(range(min(64, len(dataset)))))
    loader = DataLoader(dataset, batch_size=cfg.per_gpu_batch_size, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0, drop_last=True)
    val_loader = None
    if (Path(cfg.split_root) / "val.txt").exists():
        val_loader = DataLoader(PairedSEN12(Path(cfg.split_root) / "val.txt", False, cfg.image_size), batch_size=cfg.per_gpu_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0, drop_last=True)
    model = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim, cfg.in_chans).to(device)
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    ema = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim, cfg.in_chans).to(device); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device)
    sar_ids = {id(p) for p in model.sar_encoder.vit.parameters()}
    groups = [{"params": [p for p in model.parameters() if id(p) not in sar_ids], "lr": cfg.lr_new, "weight_decay": 0.0}]
    groups += llrd_groups(model.sar_encoder.vit, cfg.lr_sar, .75, .05)
    opt = torch.optim.AdamW(groups, betas=(.9, .95), fused=device.type == "cuda")
    logs = run / "logs.csv"
    if not logs.exists(): logs.write_text("step,l_flow,l_align,lr,it_s\n")
    step = 0; iterator = iter(loader); started = time.time(); model.train()
    resume = latest_checkpoint(run)
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"]); opt.load_state_dict(state["opt"]); step = int(state["step"]); print("resumed", resume, "at", step)
    while step < cfg.total_steps:
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        sar, eo = batch["sar"].to(device, non_blocking=True), batch["eo"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.no_grad(): x0, dino = targets.encode(eo, cfg.vae_scale)
            t = torch.rand(x0.shape[0], device=device); eps = torch.randn_like(x0); xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
            v, z = model(xt, t, sar, torch.rand(x0.shape[0], device=device) < cfg.cond_dropout, sar_trainable=step >= cfg.sar_freeze_steps)
            l_flow = F.mse_loss(v.float(), (eps - x0).float()); l_align = -(F.normalize(z.float(), -1) * F.normalize(dino.float(), -1)).sum(-1).mean()
            loss = l_flow + cfg.alignment_lambda * l_align
        if not torch.isfinite(loss): raise FloatingPointError(f"NaN/Inf loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step(); opt.zero_grad(set_to_none=True)
        update_ema(ema, model, cfg.ema_decay); step += 1
        # Linear warmup applies to all optimizer groups, preserving their LR ratios.
        scale = min(1.0, step / max(1, cfg.warmup_steps))
        for group, base in zip(opt.param_groups, [cfg.lr_new] + [cfg.lr_sar] * (len(opt.param_groups) - 1)): group["lr"] = base * scale
        if step % 100 == 0:
            elapsed = max(time.time() - started, 1e-6); lr = opt.param_groups[0]["lr"]
            with logs.open("a", newline="") as f: csv.writer(f).writerow([step, l_flow.item(), l_align.item(), lr, step / elapsed])
            print(f"step={step} flow={l_flow.item():.5f} align={l_align.item():.5f}")
        if step % 2000 == 0: save_checkpoint(run / f"checkpoint_{step:07d}.pt", model, ema, opt, step)
        if step % 2000 == 0 and val_loader is not None:
            ema.eval(); vb = next(iter(val_loader)); vsar, veo = vb["sar"].to(device), vb["eo"].to(device)
            with torch.no_grad():
                vz, _ = targets.encode(veo, cfg.vae_scale); vt = torch.full((vz.shape[0],), 0.5, device=device); veps = torch.zeros_like(vz); vv, _ = ema((1 - vt[:, None, None, None]) * vz, vt, vsar, torch.zeros(vz.shape[0], device=device, dtype=torch.bool), sar_trainable=False); val_flow = F.mse_loss(vv.float(), (-vz).float()).item()
            validation_log = run / "validation.csv"
            if not validation_log.exists(): validation_log.write_text("step,l_flow\n")
            with validation_log.open("a", newline="") as f: csv.writer(f).writerow([step, val_flow])
            print(f"validation step={step} flow={val_flow:.5f}")
    save_checkpoint(run / "checkpoint_last.pt", model, ema, opt, step)
    plot_loss_curves(logs, Path("results/loss_curves.png"))


if __name__ == "__main__": main()
