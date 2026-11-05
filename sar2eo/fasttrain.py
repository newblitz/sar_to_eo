"""Fast training loop using cached VAE latents and mixed precision."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from . import fastutils as fu
from .fastdata import to_model_inputs


@dataclass
class FastCfg:
    micro_batch: int = 16
    global_batch: int = 64
    max_steps: int = 20_000
    warmup: int = 1_000
    freeze_steps: int = 2_000
    lam: float = 0.5
    drop_p: float = 0.1
    ema_decay: float = 0.999
    clip: float = 1.0
    workers: int = 2
    log_every: int = 50
    val_every: int = 500
    ckpt_every_steps: int = 500
    ckpt_every_sec: float = 900.0
    ckpt_path: str = "runs/main/last.pt"
    best_path: str = "runs/main/best_ema.pt"
    t_grid: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def make_dino_tokens(dino, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    dtype, _ = fu.pick_precision()
    dino.eval().requires_grad_(False)

    @torch.no_grad()
    def tokens(eo):
        image = F.interpolate(
            (eo + 1) / 2, size=224, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            output = dino(pixel_values=(image - mean) / std).last_hidden_state[:, 1:]
        return output.float()

    return tokens


def infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def make_loss_fn(dino_tokens, cfg, device, stash):
    def loss_fn(model, batch):
        sar, eo, x0 = to_model_inputs(*batch, device=device)
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=device)
        eps = torch.randn_like(x0)
        x_t = (1 - t).view(-1, 1, 1, 1) * x0 + t.view(-1, 1, 1, 1) * eps
        drop = torch.rand(batch_size, device=device) < cfg.drop_p
        target = dino_tokens(eo)
        velocity, aligned = model(x_t, t, sar, drop)
        flow = F.mse_loss(velocity.float(), (eps - x0).float())
        align = -(
            F.normalize(aligned.float(), dim=-1)
            * F.normalize(target.float(), dim=-1)
        ).sum(-1).mean()
        stash["flow"], stash["align"] = flow.detach(), align.detach()
        return flow + cfg.lam * align

    return loss_fn


@torch.no_grad()
def evaluate_flow(ema, val_ds, cfg, device, max_items=600, batch=16):
    ema.eval()
    dtype, _ = fu.pick_precision()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    total, count = 0.0, 0
    limit = min(len(val_ds), max_items)
    for start in range(0, limit, batch):
        items = [val_ds[index] for index in range(start, min(start + batch, limit))]
        sar, _, x0 = to_model_inputs(
            *[torch.stack(values) for values in zip(*items)], device=device
        )
        eps = torch.randn(x0.shape, generator=generator).to(device)
        for time_value in cfg.t_grid:
            t = torch.full((x0.shape[0],), time_value, device=device)
            x_t = (1 - time_value) * x0 + time_value * eps
            with torch.autocast(
                "cuda", dtype=dtype, enabled=dtype is not None
            ):
                velocity, _ = ema(x_t, t, sar)
            total += F.mse_loss(
                velocity.float(), (eps - x0).float(), reduction="sum"
            ).item()
            count += x0.numel()
    return total / count


def train_loop(
    model,
    ema,
    optimizer,
    train_ds,
    val_ds,
    dino_tokens,
    cfg,
    device="cuda",
    rank=0,
    world=1,
    log=print,
):
    dtype, use_scaler = fu.pick_precision()
    scaler = fu.make_scaler(use_scaler)
    accum = max(1, cfg.global_batch // (cfg.micro_batch * world))
    sampler = (
        DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
        if world > 1
        else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=True,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None,
    )
    if not len(loader):
        raise RuntimeError("training set is smaller than one micro-batch")
    log(
        f"micro_batch={cfg.micro_batch} accum={accum} world={world} "
        f"effective_global_batch={cfg.micro_batch * accum * world} "
        f"precision={dtype} grad_scaler={scaler.is_enabled()}"
    )
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
    step = 0
    if os.path.exists(cfg.ckpt_path):
        step = fu.load_ckpt(cfg.ckpt_path, model, ema, optimizer, scaler, "cpu")
        log(f"resumed from {cfg.ckpt_path} at step {step}")

    model_parameters = list(_unwrap(model).parameters())
    ema_parameters = list(ema.parameters())
    stash = {}
    loss_fn = make_loss_fn(dino_tokens, cfg, device, stash)
    batches = infinite(loader, sampler)
    best = float("inf")
    started = time.perf_counter()
    last_checkpoint = started
    timed_steps = 0
    model.train()
    while step < cfg.max_steps:
        _unwrap(model).sar_trainable = step >= cfg.freeze_steps
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * min(1.0, (step + 1) / cfg.warmup)
        micro_batches = [next(batches) for _ in range(accum)]
        loss = fu.accumulate_step(model, micro_batches, loss_fn, scaler, dtype)
        fu.optimizer_step(model, optimizer, scaler, cfg.clip)
        with torch.no_grad():
            torch._foreach_lerp_(ema_parameters, model_parameters, 1 - cfg.ema_decay)
        step += 1
        timed_steps += 1

        if rank == 0 and (step % cfg.log_every == 0 or step == cfg.max_steps):
            seconds = (time.perf_counter() - started) / timed_steps
            log(
                f"step={step} loss={float(loss):.4f} flow={float(stash['flow']):.4f} "
                f"align={float(stash['align']):.4f} {seconds:.2f}s/step "
                f"{cfg.global_batch / seconds:.1f} samples/s"
            )
            started, timed_steps = time.perf_counter(), 0

        if rank == 0 and step % cfg.val_every == 0:
            validation = evaluate_flow(ema, val_ds, cfg, device)
            log(f"validation step={step} flow={validation:.5f}")
            if validation < best:
                best = validation
                os.makedirs(os.path.dirname(os.path.abspath(cfg.best_path)), exist_ok=True)
                torch.save(ema.state_dict(), cfg.best_path)
            model.train()
        if world > 1 and dist.is_initialized() and step % cfg.val_every == 0:
            dist.barrier()

        now = time.perf_counter()
        due = (
            step % cfg.ckpt_every_steps == 0
            or now - last_checkpoint > cfg.ckpt_every_sec
            or step == cfg.max_steps
        )
        if rank == 0 and due:
            fu.save_ckpt(cfg.ckpt_path, model, ema, optimizer, scaler, step)
            last_checkpoint = now
    return step
