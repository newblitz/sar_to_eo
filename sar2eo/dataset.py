from pathlib import Path
from PIL import Image
import random
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_list(path: str | Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            parts = line.split("\t")
            if len(parts) not in (2, 3):
                raise ValueError(f"Expected SAR<TAB>EO[<TAB>category] row: {line}")
            rows.append((parts[0], parts[1], parts[2] if len(parts) == 3 else "unknown"))
    return rows


class PairedSEN12(Dataset[dict[str, Any]]):
    """Reads split rows as ``sar_path<TAB>eo_path`` and applies joint geometry."""
    def __init__(self, split_file: str | Path, augment: bool = False, image_size: int = 256):
        self.rows = _read_list(split_file)
        self.augment, self.image_size = augment, image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sar_path, eo_path, category = self.rows[index]
        sar = Image.open(sar_path).convert("L")
        eo = Image.open(eo_path).convert("RGB")
        if sar.size != (self.image_size, self.image_size) or eo.size != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} pair: {sar_path}, {eo_path}")
        if self.augment:
            if random.random() < 0.5:
                sar, eo = sar.transpose(Image.Transpose.FLIP_LEFT_RIGHT), eo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                sar, eo = sar.transpose(Image.Transpose.FLIP_TOP_BOTTOM), eo.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randrange(4)
            if turns:
                sar, eo = sar.rotate(90 * turns), eo.rotate(90 * turns)
        s = torch.from_numpy(np.asarray(sar, dtype=np.float32)).div(127.5).sub(1).unsqueeze(0)
        e = torch.from_numpy(np.asarray(eo, dtype=np.float32)).div(127.5).sub(1).permute(2, 0, 1)
        return {"sar": s, "eo": e, "sar_path": sar_path, "eo_path": eo_path, "category": category}
