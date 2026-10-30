from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    image_size: int = 256
    latent_size: int = 32
    in_chans: int = 1
    latent_chans: int = 4
    d_model: int = 768
    depth: int = 12
    heads: int = 12
    align_layer: int = 4
    dino_dim: int = 768
    cond_dropout: float = 0.1
    alignment_lambda: float = 0.5
    global_batch_size: int = 64
    per_gpu_batch_size: int = 1
    grad_accum_steps: int = 1
    num_workers: int = 8
    lr_new: float = 2e-4
    lr_backbone: float = 2e-5
    lr_sar: float = 2e-5
    warmup_steps: int = 1000
    sar_freeze_steps: int = 3000
    total_steps: int = 60_000
    ema_decay: float = 0.999
    seed: int = 0
    grad_clip: float = 1.0
    vae_scale: float = 0.18215
    data_root: str = "v_2"
    output_root: str = "runs/main"
    split_root: str = "splits"
    device: str = "auto"
    vae_id: str = "assets/sd-vae-ft-ema"
    dino_id: str = "assets/dinov2-base"
    sit_checkpoint: str = ""
    sar_checkpoint: str = "/kaggle/input/datasets/prateeking/sar-jepa-input/checkpoint-200.pth"
    sar_preprocessing: str = "8-bit scaled intensity; x/127.5 - 1; no dB conversion"
    terrain_categories: tuple[str, ...] = ("agriculture", "barren", "grassland", "urban")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))

    @property
    def is_debug(self) -> bool:
        return self.d_model == 768 and self.depth == 12
