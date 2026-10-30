"""Run one real two-GPU pipeline step without creating a training run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .configs import Config
    from .dataset import PairedSEN12
    from .frozen import FrozenTargets
    from .model import SAR2EO
    from .sar_encoder import load_sar_checkpoint
except ImportError:
    from sar2eo.configs import Config
    from sar2eo.dataset import PairedSEN12
    from sar2eo.frozen import FrozenTargets
    from sar2eo.model import SAR2EO
    from sar2eo.sar_encoder import load_sar_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Optional saved Config JSON.")
    parser.add_argument("--split", default="splits/train.txt")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Launch this smoke test with torchrun --nproc_per_node=2.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(f"Expected {world_size} CUDA devices, found {torch.cuda.device_count()}.")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(
            backend="nccl", init_method="env://", device_id=device
        )
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://")

    try:
        cfg = Config.load(args.config) if args.config else Config()
        split = Path(args.split)
        if not split.exists():
            raise FileNotFoundError(f"Missing split file: {split}")

        dataset = PairedSEN12(split, augment=False, image_size=cfg.image_size)
        if not dataset:
            raise RuntimeError(f"Split has no rows: {split}")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        batch = next(iter(loader))
        sar = batch["sar"].to(device, non_blocking=True)
        eo = batch["eo"].to(device, non_blocking=True)

        # Adapt the model before moving it to CUDA or constructing DDP. This is
        # essential for the official SAR-JEPA iRPE encoder replacement.
        base_model = SAR2EO(
            cfg.d_model,
            cfg.depth,
            cfg.heads,
            cfg.align_layer,
            cfg.dino_dim,
            cfg.in_chans,
        )
        load_sar_checkpoint(base_model.sar_encoder, cfg.sar_checkpoint)
        base_model = base_model.to(device)
        model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        targets = FrozenTargets(cfg.vae_id, cfg.dino_id).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_new, betas=(0.9, 0.95))
        model.train()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                x0, dino = targets.encode(eo, cfg.vae_scale)
            t = torch.full((x0.shape[0],), 0.5, device=device)
            eps = torch.randn_like(x0)
            xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
            velocity, aligned = model(
                xt,
                t,
                sar,
                torch.zeros(x0.shape[0], dtype=torch.bool, device=device),
                sar_trainable=True,
            )
            flow_loss = F.mse_loss(velocity.float(), (eps - x0).float())
            align_loss = -(
                F.normalize(aligned.float(), dim=-1) * F.normalize(dino.float(), dim=-1)
            ).sum(-1).mean()
            loss = flow_loss + cfg.alignment_lambda * align_loss

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Non-finite smoke-test loss: {loss.item()}")
        loss.backward()
        trainable_grads = [
            parameter.grad for parameter in model.parameters() if parameter.requires_grad
        ]
        if not any(gradient is not None and torch.isfinite(gradient).all() for gradient in trainable_grads):
            raise RuntimeError("No finite gradients were produced for trainable model parameters.")
        if any(parameter.grad is not None for parameter in targets.parameters()):
            raise RuntimeError("A frozen VAE or DINOv2 parameter received a gradient.")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.barrier()

        if rank == 0:
            print(
                {
                    "status": "PASS",
                    "world_size": world_size,
                    "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                    "sar": tuple(sar.shape),
                    "eo": tuple(eo.shape),
                    "latent": tuple(x0.shape),
                    "dino": tuple(dino.shape),
                    "velocity": tuple(velocity.shape),
                    "aligned": tuple(aligned.shape),
                    "loss": float(loss.detach().item()),
                    "optimizer_step": "completed",
                },
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
