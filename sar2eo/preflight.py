"""Offline asset, package, and hardware preflight for the complete run."""
from pathlib import Path
import argparse, importlib.util, subprocess, sys
import torch


def torchvision_compat():
    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        lib.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        return lib
    except Exception:
        return None


def version(name):
    try: return getattr(__import__(name), "__version__", "installed")
    except Exception: return "missing"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args(); root = Path(a.root)
    _tv = torchvision_compat()
    print("python:", sys.version); print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0), "memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "bf16:", torch.cuda.is_bf16_supported())
    for package in ["diffusers", "transformers", "huggingface_hub", "timm", "torchmetrics", "lpips", "cv2"]: print(package + ":", version(package))
    assets = {"SiT source": root / "assets/SiT", "SAR-JEPA source": root / "assets/SAR-JEPA", "SD-VAE": root / "assets/sd-vae-ft-ema", "DINOv2": root / "assets/dinov2-base", "SAR-JEPA checkpoint": root / "assets/sar_jepa"}
    for name, path in assets.items(): print(name + ":", path, "exists=" + str(path.exists()))
    attached = root / "sar-jeppa/checkpoint-200.pth"; print("attached SAR-JEPA checkpoint:", attached, "exists=" + str(attached.exists()))
    if attached.exists():
        try: from .sar_encoder import inspect_checkpoint
        except ImportError: from sar2eo.sar_encoder import inspect_checkpoint
        inspect_checkpoint(attached, root / "results/sar_jepa_checkpoint_report.txt")
    for path in [root / "assets/sd-vae-ft-ema/config.json", root / "assets/dinov2-base/config.json"]: print(path, "exists=" + str(path.exists()))
    for path in [root / "splits/train.txt", root / "splits/val.txt", root / "splits/test.txt"]: print(path, "exists=" + str(path.exists()))


if __name__ == "__main__": main()
