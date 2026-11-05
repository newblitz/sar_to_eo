"""Train SAR-to-EO from cached images and VAE latents."""

from __future__ import annotations

import argparse
import copy
import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Dinov2Model

try:
    from .configs import Config
    from .fastdata import FastPairs
    from .fasttrain import FastCfg, make_dino_tokens, train_loop
    from .model import SAR2EO
    from .sar_encoder import llrd_groups, load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.fastdata import FastPairs
    from sar2eo.fasttrain import FastCfg, make_dino_tokens, train_loop
    from sar2eo.model import SAR2EO
    from sar2eo.sar_encoder import llrd_groups, load_sar_checkpoint


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                "nccl",
                init_method="env://",
                timeout=datetime.timedelta(minutes=60),
                device_id=torch.device("cuda", local_rank),
            )
        except TypeError:
            dist.init_process_group(
                "nccl", init_method="env://", timeout=datetime.timedelta(minutes=60)
            )
        rank = dist.get_rank()
    else:
        rank = 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return distributed, world, rank, local_rank, device


def build_optimizer(model, cfg, device):
    sar_ids = {id(parameter) for parameter in model.sar_encoder.vit.parameters()}
    groups = [
        {
            "params": [p for p in model.parameters() if id(p) not in sar_ids],
            "lr": cfg.lr_new,
            "weight_decay": 0.0,
        }
    ]
    groups += llrd_groups(model.sar_encoder.vit, cfg.lr_sar, 0.75, 0.05)
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", default=os.environ.get("CACHE_DIR", "/kaggle/working/cache"))
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()

    distributed, world, rank, local_rank, device = setup_distributed()
    is_main = rank == 0
    cfg = Config.load(args.config) if args.config else Config(output_root=f"runs/{args.run}")
    run = "overfit" if args.overfit else args.run
    max_steps = 300 if args.overfit else args.steps
    cache = args.cache_dir
    train_prefix = os.path.join(cache, "train")
    val_prefix = os.path.join(cache, "val")
    train_lat = train_prefix + "_lat.npy"
    val_lat = val_prefix + "_lat.npy"
    required = (train_prefix + "_sar.npy", train_prefix + "_eo.npy", train_lat,
                val_prefix + "_sar.npy", val_prefix + "_eo.npy", val_lat)
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Training caches are missing. Run scripts/make_cache.py first: "
            + ", ".join(missing)
        )

    torch.manual_seed(cfg.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    if distributed:
        dist.barrier()

    train_ds = FastPairs(train_prefix, train_lat, train=True)
    val_ds = FastPairs(val_prefix, val_lat, train=False)
    if args.overfit:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = Subset(val_ds, range(min(64, len(val_ds))))

    model = SAR2EO(
        cfg.d_model, cfg.depth, cfg.heads, cfg.align_layer, cfg.dino_dim, cfg.in_chans
    ).to(device)
    load_sar_checkpoint(model.sar_encoder, cfg.sar_checkpoint)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = build_optimizer(model, cfg, device)
    model_for_train = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    ) if distributed else model
    dino = Dinov2Model.from_pretrained(cfg.dino_id, local_files_only=True).to(device)
    dino_tokens = make_dino_tokens(dino, device)

    output_root = os.path.join("/kaggle/working/runs", run) if args.cache_dir.startswith("/kaggle") else cfg.output_root
    fast_cfg = FastCfg(
        micro_batch=args.micro_batch,
        global_batch=cfg.global_batch_size,
        max_steps=max_steps,
        warmup=50 if args.overfit else cfg.warmup_steps,
        freeze_steps=0 if args.overfit else 2_000,
        lam=cfg.alignment_lambda,
        drop_p=cfg.cond_dropout,
        ema_decay=0.99 if args.overfit else cfg.ema_decay,
        workers=2,
        val_every=100 if args.overfit else 500,
        ckpt_every_steps=100 if args.overfit else 500,
        ckpt_path=os.path.join(output_root, "last.pt"),
        best_path=os.path.join(output_root, "best_ema.pt"),
    )
    if is_main:
        os.makedirs(output_root, exist_ok=True)
        cfg.save(os.path.join(output_root, "config.json"))
        print(f"device={device} cache={cache} run={output_root}")
    train_loop(
        model_for_train, ema, optimizer, train_ds, val_ds, dino_tokens,
        fast_cfg, device, rank, world
    )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
