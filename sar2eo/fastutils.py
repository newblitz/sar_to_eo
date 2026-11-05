"""Precision, accumulation, checkpointing, PSNR, and benchmark helpers."""

import gc
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch


def pick_precision():
    if not torch.cuda.is_available():
        return None, False
    major, _ = torch.cuda.get_device_capability()
    return (torch.bfloat16, False) if major >= 8 else (torch.float16, True)


def make_scaler(use_scaler):
    return torch.amp.GradScaler(
        "cuda", enabled=bool(use_scaler) and torch.cuda.is_available()
    )


def accumulate_step(model, batches, loss_fn, scaler=None, autocast_dtype=None):
    total = 0
    ddp = hasattr(model, "no_sync")
    for index, batch in enumerate(batches):
        context = model.no_sync() if ddp and index < len(batches) - 1 else nullcontext()
        with context:
            with torch.autocast(
                "cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                loss = loss_fn(model, batch) / len(batches)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        total = total + loss.detach()
    return total


def optimizer_step(model, optimizer, scaler=None, clip=1.0):
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def find_micro_batch(run_one, candidates=(16, 8, 4, 2, 1)):
    for batch_size in candidates:
        try:
            run_one(batch_size)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return batch_size
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("even micro-batch 1 runs out of memory")


def psnr01(a, b):
    a, b = (a.clamp(-1, 1) + 1) / 2, (b.clamp(-1, 1) + 1) / 2
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_ckpt(path, model, ema, optimizer, scaler, step, extra=None):
    state = {
        "model": _unwrap(model).state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "rng": {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_ckpt(path, model, ema, optimizer, scaler, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    _unwrap(model).load_state_dict(state["model"])
    if ema is not None and state["ema"] is not None:
        ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["opt"])
    if scaler is not None and state["scaler"]:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["py"])
    np.random.set_state(state["rng"]["np"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return int(state["step"])


def benchmark(step_fn, warmup=5, iters=20):
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters
