"""Compatibility helpers for optional torchvision imports."""

import importlib.util
import sys
from pathlib import Path
import torch


_TV_LIB = None


def _operator_exists(name: str) -> bool:
    try:
        getattr(torch.ops.torchvision, name)
        return True
    except (AttributeError, RuntimeError):
        return False


def ensure_torchvision_ops() -> None:
    """Import torchvision when possible and add only genuinely missing schemas.

    Some CPU torchvision builds omit NMS while torchvision's Python meta
    registration expects it. Other builds already register NMS in their C++
    extension. Defining the schema before that extension loads causes duplicate
    registration, so the healthy import path is always attempted first.
    """
    global _TV_LIB
    if _operator_exists("nms") and _operator_exists("qnms"):
        return
    # Load the compiled extension before importing torchvision's Python module.
    # On Kaggle this lets the real C++ NMS registration happen first and avoids
    # defining a duplicate Python schema.
    spec = importlib.util.find_spec("torchvision")
    if spec and spec.origin:
        package_dir = Path(spec.origin).parent
        for extension in sorted(package_dir.glob("*_C*.so")):
            try:
                torch.ops.load_library(str(extension))
            except (OSError, RuntimeError):
                continue
            if _operator_exists("nms"):
                break
    try:
        import torchvision  # noqa: F401
        if _operator_exists("nms") and _operator_exists("qnms"):
            return
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                sys.modules.pop(name, None)
    if _operator_exists("nms") and _operator_exists("qnms"):
        return
    _TV_LIB = torch.library.Library("torchvision", "FRAGMENT")
    if not _operator_exists("nms"):
        _TV_LIB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    if not _operator_exists("qnms"):
        _TV_LIB.define("qnms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
