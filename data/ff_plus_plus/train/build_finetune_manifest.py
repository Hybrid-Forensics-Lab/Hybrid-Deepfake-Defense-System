"""
Build a balanced 2000-image manifest: 1000 real + 1000 fake (250 per forgery type).
Each real image is paired with a fake of the SAME identity AND frame index.
Real images are partitioned across methods — no real image appears twice.
"""
import re
import csv
import random
from pathlib import Path

SEED = 42
N_PER_METHOD = 250
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

TRAIN_DIR = Path(__file__).parent
REAL_DIR = TRAIN_DIR / "Real"
FAKE_BASE = TRAIN_DIR / "Fake"
OUT_MANIFEST = TRAIN_DIR / "manifest.csv"

REAL_RE = re.compile(r"^(\d+)_f(\d+)\.jpg$")
FAKE_RE = re.compile(r"^(\d+)_\d+_f(\d+)\.jpg$")


def scan_real(directory):
    """Returns dict: (source_id, frame_idx) -> filename."""
    index = {}
    for f in directory.iterdir():
        m = REAL_RE.match(f.name)
        if m:
            key = (m.group(1), m.group(2))
            index[key] = f.name
    return index


def scan_fake(directory):
    """Returns dict: (source_id, frame_idx) -> filename."""
    index = {}
    for f in directory.iterdir():
        m = FAKE_RE.match(f.name)
        if m:
            key = (m.group(1), m.group(2))
            # keep first match per key (multiple target_ids share same source+frame)
            if key not in index:
                index[key] = f.name
    return index


if __name__ == "__main__":
    random.seed(SEED)

    real_index = scan_real(REAL_DIR)
    print(f"Real images indexed: {len(real_index)}")

    # For each method, find keys that exist in both real and fake
    method_pairs = {}
    for method in METHODS:
        fake_index = scan_fake(FAKE_BASE / method)
        print(f"{method} fake images indexed: {len(fake_index)}")
        shared_keys = list(set(real_index.keys()) & set(fake_index.keys()))
        random.shuffle(shared_keys)
        method_pairs[method] = [(k, real_index[k], fake_index[k]) for k in shared_keys]
        print(f"  Matchable pairs: {len(shared_keys)}")

    # Partition real images across methods — no real used twice
    used_real = set()
    selected = []  # (real_filename, fake_filename, method)

    for method in METHODS:
        count = 0
        for key, real_fname, fake_fname in method_pairs[method]:
            if key in used_real:
                continue
            used_real.add(key)
            selected.append((real_fname, fake_fname, method))
            count += 1
            if count == N_PER_METHOD:
                break
        if count < N_PER_METHOD:
            raise RuntimeError(
                f"{method}: only {count} matchable pairs not already used. "
                "Need {N_PER_METHOD}."
            )
        print(f"{method}: selected {count} pairs")

    # Write manifest
    rows = []
    for real_fname, fake_fname, method in selected:
        rows.append({
            "image_path": f"data/ff_plus_plus/train/Real/{real_fname}",
            "label": 0,
            "forgery_type": "Real",
            "resolution": "",
            "paired_fake": f"data/ff_plus_plus/train/Fake/{method}/{fake_fname}",
            "forgery_method": method,
        })
        rows.append({
            "image_path": f"data/ff_plus_plus/train/Fake/{method}/{fake_fname}",
            "label": 1,
            "forgery_type": method,
            "resolution": "",
            "paired_fake": f"data/ff_plus_plus/train/Real/{real_fname}",
            "forgery_method": method,
        })

    # Sort: reals first then fakes, grouped by method
    rows_real = [r for r in rows if r["label"] == 0]
    rows_fake = [r for r in rows if r["label"] == 1]
    rows_sorted = rows_real + rows_fake

    fieldnames = ["image_path", "label", "forgery_type", "resolution", "paired_fake", "forgery_method"]
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"\nWrote {len(rows_sorted)} rows to {OUT_MANIFEST}")
    print(f"  Real: {len(rows_real)}, Fake: {len(rows_fake)}")
    for method in METHODS:
        mc = sum(1 for r in rows_fake if r["forgery_method"] == method)
        print(f"  {method}: {mc} fake images")
