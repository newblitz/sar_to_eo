"""Prepare the repository layout inside a Kaggle notebook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def replace_link(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        raise RuntimeError(f"Refusing to replace existing directory: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())
    print(f"linked {target} -> {source}")


def has_dataset_layout(path: Path) -> bool:
    return all((path / name).is_dir() for name in ("agri", "barrenland", "grassland", "urban"))


def find_dataset(input_root: Path) -> Path | None:
    if has_dataset_layout(input_root):
        return input_root
    for path in input_root.rglob("*"):
        if path.is_dir() and has_dataset_layout(path):
            return path
    return None


def find_checkpoint(input_root: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        if requested.exists():
            return requested
        raise FileNotFoundError(requested)
    candidates = sorted(input_root.rglob("checkpoint-200.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple checkpoint-200.pth files found. Pass --sar-jepa-input explicitly: "
            + ", ".join(map(str, candidates))
        )
    return None


def clone_if_missing(url: str, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        print(f"using existing source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--depth", "1", url, str(destination))


def download_hf(repo_id: str, destination: Path) -> None:
    if (destination / "config.json").exists():
        print(f"using existing Hugging Face asset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run("hf", "download", repo_id, "--local-dir", str(destination))


def remove_unused_torchaudio() -> None:
    # This image-only project never imports torchaudio. Kaggle can ship a
    # torchaudio wheel built for CUDA 12.x alongside PyTorch CUDA 13.x; merely
    # having the package installed makes Transformers import it while loading
    # DINOv2 and abort before the model starts.
    try:
        import importlib.util

        if importlib.util.find_spec("torchaudio") is not None:
            run(sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio")
    except (ModuleNotFoundError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--kaggle-input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--dataset-input", type=Path)
    parser.add_argument("--sar-jepa-input", type=Path)
    parser.add_argument("--allow-missing-sar", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    input_root = args.kaggle_input.resolve()
    os.chdir(root)
    remove_unused_torchaudio()
    (root / "assets").mkdir(exist_ok=True)

    dataset = args.dataset_input.resolve() if args.dataset_input else find_dataset(input_root)
    if dataset is None or not has_dataset_layout(dataset):
        raise FileNotFoundError(
            "Could not find the four terrain folders under /kaggle/input. "
            "Attach the SEN1-2 dataset or pass --dataset-input /kaggle/input/<name>."
        )
    replace_link(root / "v_2", dataset)

    clone_if_missing("https://github.com/willisma/SiT", root / "assets/SiT")
    clone_if_missing("https://github.com/waterdisappear/SAR-JEPA", root / "assets/SAR-JEPA")
    download_hf("stabilityai/sd-vae-ft-ema", root / "assets/sd-vae-ft-ema")
    download_hf("facebook/dinov2-base", root / "assets/dinov2-base")

    checkpoint = find_checkpoint(input_root, args.sar_jepa_input)
    checkpoint_target = root / "assets/sar_jepa/checkpoint-200.pth"
    if checkpoint is None:
        if not args.allow_missing_sar:
            raise FileNotFoundError(
                "SAR-JEPA checkpoint was not found under /kaggle/input. "
                "Attach it as a Kaggle Dataset/Model or pass --sar-jepa-input."
            )
        print("WARNING: SAR-JEPA checkpoint missing; the code will report this explicitly.")
    else:
        replace_link(checkpoint_target, checkpoint)

    print("\nKaggle project layout is ready:")
    for path in (
        root / "v_2",
        root / "assets/SiT",
        root / "assets/SAR-JEPA",
        root / "assets/sd-vae-ft-ema",
        root / "assets/dinov2-base",
        root / "assets/sar_jepa/checkpoint-200.pth",
    ):
        print(f"  {path}: {'OK' if path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
