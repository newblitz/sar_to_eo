"""Offline asset, package, and hardware preflight for the complete run."""
from pathlib import Path
import argparse, importlib.util, subprocess, sys
import torch
try:
    from .compat import ensure_torchvision_ops
except ImportError:
    from sar2eo.compat import ensure_torchvision_ops


def version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args(); root = Path(a.root)
    ensure_torchvision_ops()
    print("python:", sys.version); print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0), "memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), "bf16:", torch.cuda.is_bf16_supported())
    for package in ["diffusers", "transformers", "huggingface_hub", "timm", "cv2"]:
        print(package + ":", version(package) or "missing")
    required = {
        "lpips": "lpips",
        "torchmetrics": "torchmetrics",
        "torch-fidelity": "torch_fidelity",
    }
    missing = []
    for label, module_name in required.items():
        installed = version(module_name)
        print(label + ":", "OK" if installed else "missing")
        if not installed:
            missing.append(label)
    if missing:
        raise RuntimeError(
            "Required evaluation dependencies are missing: " + ", ".join(missing)
        )
    assets = {"SiT source": root / "assets/SiT", "SAR-JEPA source": root / "assets/SAR-JEPA", "SD-VAE": root / "assets/sd-vae-ft-ema", "DINOv2": root / "assets/dinov2-base", "SAR-JEPA checkpoint": root / "assets/sar_jepa"}
    for name, path in assets.items(): print(name + ":", path, "exists=" + str(path.exists()))
    attached = root / "assets/sar_jepa/checkpoint-200.pth"
    legacy_attached = root / "sar-jeppa/checkpoint-200.pth"
    print("staged SAR-JEPA checkpoint:", attached, "exists=" + str(attached.exists()))
    if legacy_attached.exists():
        print("legacy SAR-JEPA checkpoint:", legacy_attached, "exists=True")
    if attached.exists():
        try: from .sar_encoder import inspect_checkpoint
        except ImportError: from sar2eo.sar_encoder import inspect_checkpoint
        inspect_checkpoint(attached, root / "results/sar_jepa_checkpoint_report.txt")
    for path in [root / "assets/sd-vae-ft-ema/config.json", root / "assets/dinov2-base/config.json"]: print(path, "exists=" + str(path.exists()))
    for path in [root / "splits/train.txt", root / "splits/val.txt", root / "splits/test.txt"]: print(path, "exists=" + str(path.exists()))


if __name__ == "__main__": main()
