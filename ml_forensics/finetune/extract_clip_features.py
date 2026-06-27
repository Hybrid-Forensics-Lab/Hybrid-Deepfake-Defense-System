import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import open_clip

# open_clip uses "ViT-L-14" internally; expose the familiar CLIP name to callers
MODEL_NAME_MAP = {
    "ViT-L/14": "ViT-L-14",
}


def load_model(model_arg: str, device: torch.device):
    arch = MODEL_NAME_MAP.get(model_arg, model_arg)
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
    model.eval()
    model.to(device)
    return model, preprocess


def extract_features(model, preprocess, manifest_path: Path, repo_root: Path,
                     device: torch.device, batch_size: int):
    df = pd.read_csv(manifest_path)

    all_features = []
    all_labels = []
    all_paths = []
    skipped = 0

    total = len(df)
    with tqdm(total=total, desc="Extracting CLIP features") as pbar:
        for batch_start in range(0, total, batch_size):
            batch_rows = df.iloc[batch_start: batch_start + batch_size]

            tensors = []
            batch_paths = []
            batch_labels = []

            for _, row in batch_rows.iterrows():
                img_path = repo_root / row["image_path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensors.append(preprocess(img))
                    batch_paths.append(row["image_path"])
                    batch_labels.append(int(row["label"]))
                except Exception as e:
                    print(f"\nSkipping {img_path}: {e}")
                    skipped += 1
                    pbar.update(1)
                    continue

            if not tensors:
                pbar.update(len(batch_rows))
                continue

            batch_tensor = torch.stack(tensors).to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    features = model.encode_image(batch_tensor)

            # Convert to float32 for sklearn compatibility
            all_features.append(features.float().cpu().numpy())
            all_labels.extend(batch_labels)
            all_paths.extend(batch_paths)

            pbar.update(len(batch_rows))

    if skipped:
        print(f"Warning: skipped {skipped} images due to load errors.")

    return (
        np.concatenate(all_features, axis=0),
        np.array(all_labels, dtype=np.int32),
        np.array(all_paths, dtype=object),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frozen CLIP features from a manifest CSV.")
    parser.add_argument("--manifest",   required=True,  help="Path to manifest CSV (relative to repo root or absolute)")
    parser.add_argument("--output",     required=True,  help="Output .npz file path")
    parser.add_argument("--model",      default="ViT-L/14", choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repo_root",  default=None,
                        help="Repo root. Defaults to two levels above this script.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Manifest: {manifest_path} ({pd.read_csv(manifest_path).shape[0]} images)")
    print(f"Output: {output_path}")

    model, preprocess = load_model(args.model, device)
    print(f"Loaded {args.model} via open_clip (frozen, no grad).")

    features, labels, paths = extract_features(
        model, preprocess, manifest_path, repo_root, device, args.batch_size
    )

    torch.cuda.empty_cache()

    np.savez(output_path, features=features, labels=labels, paths=paths)

    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()
    print(f"Saved {len(features)} feature vectors ({n_real} real, {n_fake} fake) "
          f"of shape {features.shape[1]}-d → {output_path}")
