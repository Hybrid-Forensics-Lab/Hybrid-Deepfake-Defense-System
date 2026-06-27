import argparse
import numpy as np
import pandas as pd
from pathlib import Path

FAKE_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def build_manifest(repo_root: Path, n_per_class: int, seed: int) -> pd.DataFrame:
    train_dir = repo_root / "data" / "ff_plus_plus" / "train"
    existing_manifest = train_dir / "finetune_manifest.csv"
    excluded_paths = set(pd.read_csv(existing_manifest)["image_path"].tolist())

    # Scan real images from disk
    real_dir = train_dir / "Real"
    real_rows = [
        {"image_path": f"data/ff_plus_plus/train/Real/{p.name}", "label": 0}
        for p in sorted(real_dir.glob("*.jpg"))
    ]

    # Scan fake images from disk across available fake type directories
    fake_rows = []
    for ft in FAKE_TYPES:
        fake_dir = train_dir / "Fake" / ft
        if not fake_dir.exists():
            continue
        for p in sorted(fake_dir.glob("*.jpg")):
            fake_rows.append({
                "image_path": f"data/ff_plus_plus/train/Fake/{ft}/{p.name}",
                "label": 1,
                "fake_type": ft,
            })

    real_df = pd.DataFrame(real_rows)
    fake_df = pd.DataFrame(fake_rows)

    # Exclude images already in the 2k finetune manifest
    real_df = real_df[~real_df["image_path"].isin(excluded_paths)].reset_index(drop=True)
    fake_df = fake_df[~fake_df["image_path"].isin(excluded_paths)].reset_index(drop=True)

    print(f"Available on disk after exclusion: {len(real_df)} real, {len(fake_df)} fake")
    for ft in FAKE_TYPES:
        count = (fake_df["fake_type"] == ft).sum() if "fake_type" in fake_df.columns else 0
        print(f"  {ft}: {count}")

    # Sample real images
    n_real = min(n_per_class, len(real_df))
    real_sample = real_df.sample(n=n_real, random_state=seed)[["image_path", "label"]]

    # Sample fake images: stratified across fake types
    n_per_type = n_per_class // len(FAKE_TYPES)
    remainder = n_per_class - n_per_type * len(FAKE_TYPES)
    fake_parts = []
    for i, ft in enumerate(FAKE_TYPES):
        sub = fake_df[fake_df["fake_type"] == ft]
        n = n_per_type + (1 if i < remainder else 0)
        n = min(n, len(sub))
        fake_parts.append(sub.sample(n=n, random_state=seed)[["image_path", "label"]])

    fake_sample = pd.concat(fake_parts)

    manifest = pd.concat([real_sample, fake_sample]) \
        .sample(frac=1, random_state=seed).reset_index(drop=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a balanced 10k finetune manifest from files on disk.")
    parser.add_argument("--output",      required=True,  help="Output CSV path")
    parser.add_argument("--n_per_class", type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--repo_root",   default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(repo_root, args.n_per_class, args.seed)
    manifest.to_csv(output_path, index=False)

    n_real = (manifest["label"] == 0).sum()
    n_fake = (manifest["label"] == 1).sum()
    print(f"Saved {len(manifest)} rows ({n_real} real, {n_fake} fake) → {output_path}")
