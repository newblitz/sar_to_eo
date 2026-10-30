import argparse
import contextlib
import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

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


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        rank_device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(
                backend="nccl", init_method="env://", device_id=rank_device
            )
        except TypeError:
            # PyTorch before device_id support still uses the device set above.
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        local_rank, rank = 0, 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return distributed, world_size, rank, local_rank, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def update_ema(ema, model, decay):
    with torch.no_grad():
        for e, p in zip(ema.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - decay)


def save_checkpoint(path, model, ema, opt, step):
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "rng": {"torch": torch.get_rng_state(), "python": random.getstate()},
        },
        path,
    )


def latest_checkpoint(run: Path):
    paths = sorted(run.glob("checkpoint_*.pt"))
    last = run / "checkpoint_last.pt"
    if last.exists():
        paths.append(last)
    return paths[-1] if paths else None


def plot_loss_curves(logs, output):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.loadtxt(logs, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[None, :]
        plt.plot(data[:, 0], data[:, 1], label="flow")
        plt.plot(data[:, 0], data[:, 2], label="alignment")
        plt.xlabel("step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output)
        plt.close()
    except Exception as exc:
        print("loss curve not written:", exc)


def mean_across_workers(value, distributed, device):
    result = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--config")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--overfit", action="store_true")
    args = ap.parse_args()

    distributed, world_size, rank, local_rank, device = setup_distributed()
    is_main = rank == 0
    cfg = Config.load(args.config) if args.config else Config(output_root=f"runs/{args.run}")
    if args.overfit:
        cfg.total_steps = 1000
    if args.steps:
        cfg.total_steps = args.steps

    random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    run = Path(cfg.output_root)
    run.mkdir(parents=True, exist_ok=True)
    if is_main:
        cfg.save(run / "config.json")
    if distributed:
        dist.barrier()

    split = Path(cfg.split_root) / "train.txt"
    if not split.exists():
        raise FileNotFoundError(f"Missing {split}; provide the dataset and run prepare_data.py.")

    dataset = PairedSEN12(split, augment=True, image_size=cfg.image_size)
    if args.overfit:
        dataset = Subset(dataset, list(range(min(64, len(dataset)))))
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.per_gpu_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )

    val_loader = None
    val_sampler = None
    val_path = Path(cfg.split_root) / "val.txt"
    if val_path.exists():
        val_dataset = PairedSEN12(val_path, False, cfg.image_size)
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.per_gpu_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
            drop_last=True,
        )

    base_model = SAR2EO(cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim, cfg.in_chans)
    load_sar_checkpoint(base_model.sar_encoder, cfg.sar_checkpoint)
    base_model = base_model.to(device)
    # Copy after SAR-JEPA adaptation. Constructing a second SAR2EO independently
    # can select the timm ViT path, while the loaded base model uses the compact
    # SAR-JEPA-compatible encoder with different key names and positional shape.
    ema = copy.deepcopy(base_model).to(device)
    ema.requires_grad_(False)

    sar_ids = {id(p) for p in base_model.sar_encoder.vit.parameters()}
    groups = [
        {
            "params": [p for p in base_model.parameters() if id(p) not in sar_ids],
            "lr": cfg.lr_new,
            "weight_decay": 0.0,
        }
    ]
    groups += llrd_groups(base_model.sar_encoder.vit, cfg.lr_sar, 0.75, 0.05)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=device.type == "cuda")

    resume = latest_checkpoint(run)
    step = 0
    micro_step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        base_model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        if is_main:
            print("resumed", resume, "at", step)

    model = base_model
    if distributed:
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device)
    logs = run / "logs.csv"
    if is_main and not logs.exists():
        logs.write_text("step,l_flow,l_align,lr,it_s\n")

    auto_accum = math.ceil(cfg.global_batch_size / max(1, cfg.per_gpu_batch_size * world_size))
    accum_steps = max(1, cfg.grad_accum_steps, auto_accum)
    if is_main:
        print(
            f"distributed={distributed} world_size={world_size} device={device} "
            f"per_gpu_batch={cfg.per_gpu_batch_size} grad_accum={accum_steps} "
            f"effective_global_batch={cfg.per_gpu_batch_size * world_size * accum_steps}"
        )

    started = time.time()
    iterator = iter(loader)
    model.train()
    opt.zero_grad(set_to_none=True)

    while step < cfg.total_steps:
        if sampler is not None:
            sampler.set_epoch(micro_step // max(1, len(loader)))
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        sar = batch["sar"].to(device, non_blocking=True)
        eo = batch["eo"].to(device, non_blocking=True)
        micro_step += 1
        sync = micro_step % accum_steps == 0
        sync_context = model.no_sync() if distributed and not sync else contextlib.nullcontext()
        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                with torch.no_grad():
                    x0, dino = targets.encode(eo, cfg.vae_scale)
                t = torch.rand(x0.shape[0], device=device)
                eps = torch.randn_like(x0)
                xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
                v, z = model(
                    xt,
                    t,
                    sar,
                    torch.rand(x0.shape[0], device=device) < cfg.cond_dropout,
                    sar_trainable=step >= cfg.sar_freeze_steps,
                )
                l_flow = F.mse_loss(v.float(), (eps - x0).float())
                l_align = -(
                    F.normalize(z.float(), -1) * F.normalize(dino.float(), -1)
                ).sum(-1).mean()
                loss = (l_flow + cfg.alignment_lambda * l_align) / accum_steps
            finite = torch.isfinite(loss.detach())
            if distributed:
                finite_int = finite.to(torch.int32)
                dist.all_reduce(finite_int, op=dist.ReduceOp.MIN)
                finite = finite_int.bool()
            if not finite:
                raise FloatingPointError(f"NaN/Inf loss at step {step}")
            loss.backward()

        if sync:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            update_ema(ema, base_model, cfg.ema_decay)
            step += 1
            scale = min(1.0, step / max(1, cfg.warmup_steps))
            bases = [cfg.lr_new] + [cfg.lr_sar] * (len(opt.param_groups) - 1)
            for group, base in zip(opt.param_groups, bases):
                group["lr"] = base * scale

            flow_value = mean_across_workers(l_flow.item(), distributed, device)
            align_value = mean_across_workers(l_align.item(), distributed, device)
            if is_main and step % 100 == 0:
                elapsed = max(time.time() - started, 1e-6)
                lr = opt.param_groups[0]["lr"]
                with logs.open("a", newline="") as f:
                    csv.writer(f).writerow([step, flow_value, align_value, lr, step / elapsed])
                print(
                    f"step={step} flow={flow_value:.5f} align={align_value:.5f} "
                    f"lr={lr:.3e}"
                )

            if step % 2000 == 0:
                if distributed:
                    dist.barrier()
                if is_main:
                    save_checkpoint(run / f"checkpoint_{step:07d}.pt", base_model, ema, opt, step)
                if distributed:
                    dist.barrier()

            if step % 2000 == 0 and val_loader is not None:
                ema.eval()
                val_iter = iter(val_loader)
                try:
                    vb = next(val_iter)
                except StopIteration:
                    vb = None
                if vb is not None:
                    vsar = vb["sar"].to(device, non_blocking=True)
                    veo = vb["eo"].to(device, non_blocking=True)
                    with torch.no_grad():
                        vz, _ = targets.encode(veo, cfg.vae_scale)
                        vt = torch.full((vz.shape[0],), 0.5, device=device)
                        vv, _ = ema(
                            (1 - vt[:, None, None, None]) * vz,
                            vt,
                            vsar,
                            torch.zeros(vz.shape[0], device=device, dtype=torch.bool),
                            sar_trainable=False,
                        )
                        local_val_flow = F.mse_loss(vv.float(), (-vz).float()).item()
                    val_flow = mean_across_workers(local_val_flow, distributed, device)
                    if is_main:
                        validation_log = run / "validation.csv"
                        if not validation_log.exists():
                            validation_log.write_text("step,l_flow\n")
                        with validation_log.open("a", newline="") as f:
                            csv.writer(f).writerow([step, val_flow])
                        print(f"validation step={step} flow={val_flow:.5f}")
                ema.train()

    if distributed:
        dist.barrier()
    if is_main:
        save_checkpoint(run / "checkpoint_last.pt", base_model, ema, opt, step)
        plot_loss_curves(logs, Path("results/loss_curves.png"))
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
