"""Inspect v_2 and create balanced, terrain-stratified SEN1-2 splits."""
from collections import defaultdict
from pathlib import Path
import argparse, json, random, re
import numpy as np
from PIL import Image, ImageDraw

CATEGORIES = {"agri": "agriculture", "barrenland": "barren", "grassland": "grassland", "urban": "urban"}
TARGETS = {"train": 3500, "val": 150, "test": 350}


def pair_category(root: Path, folder: str):
    sar_dir, eo_dir = root / folder / "s1", root / folder / "s2"
    sar_files, eo_files = sorted(sar_dir.glob("*.png")), {p.name: p for p in eo_dir.glob("*.png")}
    if len(sar_files) != 4000 or len(eo_files) != 4000:
        raise AssertionError(f"{folder}: expected 4000/4000 files, got {len(sar_files)}/{len(eo_files)}")
    rows = []
    for sar in sar_files:
        eo_name = sar.name.replace("_s1_", "_s2_")
        eo = eo_files.get(eo_name)
        if eo is None: raise AssertionError(f"Missing EO partner for {sar}")
        if sar.stem.replace("_s1_", "_s2_") != eo.stem: raise AssertionError(f"Bad pair names: {sar}, {eo}")
        if Image.open(sar).size != Image.open(eo).size: raise AssertionError(f"Size mismatch: {sar}, {eo}")
        s = np.asarray(Image.open(sar).convert("RGB"), dtype=np.float32)
        if not (np.allclose(s[..., 0], s[..., 1]) and np.allclose(s[..., 0], s[..., 2])):
            raise ValueError(f"SAR is genuinely 3-channel; refusing silent grayscale conversion: {sar}")
        e = np.asarray(Image.open(eo).convert("RGB"), dtype=np.float32)
        if s[..., 0].std() < 2: continue
        if e.std() < 5: continue
        m = re.match(r"ROIs(?P<roi>\d+)_(?P<season>[^_]+)_s1_(?P<scene>[^_]+)_p(?P<patch>\d+)$", sar.stem)
        scene = (m.group("season"), m.group("scene")) if m else None
        rows.append({"sar": str(sar), "eo": str(eo), "category": CATEGORIES[folder], "scene": scene})
    if len(rows) != 4000: print(f"{folder}: {len(rows)} usable pairs after quality filtering")
    return rows


def split_category(rows, seed):
    rng = random.Random(seed); groups = defaultdict(list)
    for row in rows:
        key = row["scene"] if row["scene"] is not None else (None, row["sar"])
        groups[key].append(row)
    keys = list(groups); rng.shuffle(keys); out = {k: [] for k in TARGETS}; used = set()
    for split in ("val", "test", "train"):
        for key in keys:
            if len(out[split]) >= TARGETS[split]: break
            if key in used: continue
            out[split].extend(groups[key]); used.add(key)
    if all(len(out[k]) == TARGETS[k] for k in TARGETS): return out, False
    # This attached subset has only a handful of very large scene groups, so exact
    # 3500/150/350 counts cannot coexist with whole-scene assignment. Use the
    # prescribed contiguous-name fallback and record the residual leakage risk.
    ordered = sorted(rows, key=lambda r: r["sar"])
    return {"val": ordered[:150], "test": ordered[150:500], "train": ordered[500:4000]}, True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="v_2"); ap.add_argument("--out", default="splits"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True); Path("results").mkdir(exist_ok=True)
    print("dataset root:", root.resolve()); print("folders:", sorted(p.name for p in root.iterdir() if p.is_dir()))
    all_splits = {k: [] for k in TARGETS}; summary = {"categories": {}, "dropped": {}, "scene_split_fallback": {}}
    for folder in CATEGORIES:
        print(folder, "sample:", [str(p.relative_to(root)) for p in sorted((root / folder).rglob("*.png"))[:3]])
        rows = pair_category(root, folder); splits, fallback = split_category(rows, a.seed); summary["categories"][CATEGORIES[folder]] = {k: len(v) for k, v in splits.items()}; summary["scene_split_fallback"][CATEGORIES[folder]] = fallback
        for split, values in splits.items(): all_splits[split].extend(values)
    rng = random.Random(a.seed)
    for split, rows in all_splits.items():
        rng.shuffle(rows)
        (out / f"{split}.txt").write_text("\n".join(f"{r['sar']}\t{r['eo']}\t{r['category']}" for r in rows) + "\n")
        print(split, len(rows), {c: sum(r["category"] == c for r in rows) for c in CATEGORIES.values()})
    total_bytes = sum(Path(r["sar"]).stat().st_size + Path(r["eo"]).stat().st_size for rows in all_splits.values() for r in rows)
    summary["bytes"] = total_bytes; summary["gib"] = total_bytes / 1024**3
    print("selected bytes:", total_bytes, "GiB:", summary["gib"])
    (out / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    preview = Image.new("RGB", (4 * 512, 4 * 256), "white"); draw = ImageDraw.Draw(preview)
    first = {c: next((r for r in all_splits["train"] if r["category"] == c), None) for c in CATEGORIES.values()}
    for i, category in enumerate(CATEGORIES.values()):
        candidates = [r for r in all_splits["train"] if r["category"] == category][:4]
        for j, r in enumerate(candidates):
            tile = Image.new("RGB", (512, 256)); tile.paste(Image.open(r["sar"]).convert("L").convert("RGB"), (0, 0)); tile.paste(Image.open(r["eo"]).convert("RGB"), (256, 0)); preview.paste(tile, ((j % 4) * 512, i * 256)); draw.text(((j % 4) * 512 + 4, i * 256 + 4), category, fill="yellow")
    preview.save("results/data_preview.png")


if __name__ == "__main__": main()
