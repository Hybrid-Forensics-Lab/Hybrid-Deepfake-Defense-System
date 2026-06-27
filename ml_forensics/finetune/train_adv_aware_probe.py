"""
Adversarial-aware MLP probe retraining for FT-UnivFD.

Combines:
  - Existing 10k FF++ features (real + fake)
  - CLIP features of cloaked LFW training images (label=0, real)
  - Optionally: CLIP features of clean LFW training images (label=0, real)
  - Optionally: CLIP features of ProGAN non-test fakes (label=1) to
    counterbalance the decision-boundary shift caused by the LFW additions.

Then retrains an MLPClassifier so the probe no longer treats adversarial
perturbation patterns as deepfake indicators.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

import open_clip

REPO_ROOT = Path(__file__).resolve().parents[2]


def fix_lfw_path(raw):
    """Convert Windows-style manifest path to absolute Path on this system."""
    p = str(raw).replace("\\", "/")
    if p.startswith("Hybrid-Deepfake-Defense-System/"):
        return REPO_ROOT / p[len("Hybrid-Deepfake-Defense-System/"):]
    if Path(p).is_absolute():
        return Path(p)
    return REPO_ROOT / p


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval().to(device)
    return model, preprocess


def extract_features_from_paths(image_paths, labels, preprocess, clip_model, device,
                                 batch_size, desc="Extracting", normalize=False):
    all_feats = []
    all_labs = []
    all_paths = []
    skipped = 0

    total = len(image_paths)
    for start in tqdm(range(0, total, batch_size), desc=desc):
        batch_paths = image_paths[start: start + batch_size]
        batch_labels = labels[start: start + batch_size]

        tensors = []
        valid_paths = []
        valid_labels = []
        for p, lbl in zip(batch_paths, batch_labels):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                valid_paths.append(str(p))
                valid_labels.append(int(lbl))
            except Exception as e:
                print(f"\nSkip {p}: {e}")
                skipped += 1

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                feats = clip_model.encode_image(batch_tensor)
        feats = feats.float()
        # NOTE: the base FF++ features (clip_features_finetune_10k.npz) are stored RAW
        # (un-normalized). L2-normalizing here while the base is raw creates a magnitude
        # shortcut that the probe latches onto. Keep raw (normalize=False) for consistency.
        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        all_feats.append(feats.cpu().numpy())
        all_labs.extend(valid_labels)
        all_paths.extend(valid_paths)

    if skipped:
        print(f"Warning: skipped {skipped} images due to load errors.")

    return (
        np.concatenate(all_feats, axis=0),
        np.array(all_labs, dtype=np.int32),
        np.array(all_paths, dtype=object),
    )


def build_cloaked_manifest(cloaked_dir, lfw_manifest_path):
    """
    For each cloaked PNG in cloaked_dir, find the corresponding clean image
    from the LFW manifest.

    Returns:
        cloaked_paths  : list of absolute Path objects (cloaked PNGs)
        clean_paths    : list of absolute Path objects (original arcface images)
    """
    lfw = pd.read_csv(lfw_manifest_path)
    # Build lookup: stem (e.g. Person_Name_0001) -> clean arcface absolute path
    stem_to_clean = {}
    for _, row in lfw.iterrows():
        clean_abs = fix_lfw_path(row["arcface_path"])
        stem_to_clean[clean_abs.stem] = clean_abs

    cloaked_paths = []
    clean_paths = []

    for png in sorted(Path(cloaked_dir).glob("*_cloaked.png")):
        orig_stem = png.stem.replace("_cloaked", "")
        if orig_stem in stem_to_clean:
            cloaked_paths.append(png)
            clean_paths.append(stem_to_clean[orig_stem])
        else:
            print(f"Warning: no clean match for {png.name}")

    return cloaked_paths, clean_paths


def find_progan_train_fakes(progan_dir, test_manifest_path, n_samples, seed):
    """
    Return up to n_samples ProGAN fake image paths that are NOT in the test manifest.
    Samples proportionally across categories.
    """
    test_df = pd.read_csv(test_manifest_path)
    test_paths = set(test_df["image_path"].tolist())

    progan_dir = Path(progan_dir)
    all_fakes = sorted(progan_dir.rglob("*.png"))
    all_fakes = [p for p in all_fakes if "1_fake" in str(p)]

    non_test = []
    for p in all_fakes:
        rel = "data/progan/" + str(p.relative_to(progan_dir))
        if rel not in test_paths:
            non_test.append(p)

    rng = np.random.default_rng(seed)
    if len(non_test) > n_samples:
        indices = rng.choice(len(non_test), size=n_samples, replace=False)
        non_test = [non_test[i] for i in sorted(indices)]

    return non_test


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial-aware MLP probe retraining."
    )
    parser.add_argument(
        "--base_features",
        type=Path,
        default=REPO_ROOT / "results/phase2/forensics/clip_features_finetune_10k.npz",
        help="10k FF++ training features .npz",
    )
    parser.add_argument(
        "--cloaked_train_dir",
        type=Path,
        default=REPO_ROOT / "results/phase2/privacy/train_pgd_run/cloaked_images",
        help="Directory with cloaked training PNGs (output of PGD script).",
    )
    parser.add_argument(
        "--lfw_manifest",
        type=Path,
        default=REPO_ROOT / "data/lfw/manifest.csv",
        help="LFW manifest to map cloaked images back to clean sources.",
    )
    parser.add_argument(
        "--progan_dir",
        type=Path,
        default=REPO_ROOT / "data/progan",
        help="ProGAN dataset root (for counterbalance fake examples).",
    )
    parser.add_argument(
        "--progan_test_manifest",
        type=Path,
        default=REPO_ROOT / "data/progan/manifest.csv",
        help="ProGAN test manifest (these images are excluded from training).",
    )
    parser.add_argument(
        "--n_progan_train",
        type=int,
        default=0,
        help="Number of non-test ProGAN fake images to add as fake training examples "
             "(counterbalance for LFW additions). 0 = disabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/phase2/forensics/mlp_probe_adv_aware.pkl",
        help="Output .pkl path for the adversarial-aware probe.",
    )
    parser.add_argument(
        "--cloaked_features_out",
        type=Path,
        default=REPO_ROOT / "results/phase2/forensics/clip_features_cloaked_lfw_train.npz",
        help="Output .npz for cloaked LFW training features.",
    )
    parser.add_argument(
        "--clean_features_out",
        type=Path,
        default=REPO_ROOT / "results/phase2/forensics/clip_features_clean_lfw_train.npz",
        help="Output .npz for clean LFW training features.",
    )
    parser.add_argument(
        "--progan_train_features_out",
        type=Path,
        default=REPO_ROOT / "results/phase2/forensics/clip_features_progan_train.npz",
        help="Output .npz for ProGAN training fake features.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_sizes", default="512,128")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_lr",
        action="store_true",
        default=False,
        help="Use LogisticRegression instead of MLPClassifier. "
             "More stable, avoids random-init sensitivity.",
    )
    parser.add_argument(
        "--lr_C",
        type=float,
        default=0.01,
        help="LogisticRegression regularization strength (smaller = more regularized). "
             "Only used when --use_lr is set.",
    )
    parser.add_argument(
        "--skip_clean",
        action="store_true",
        default=False,
        help="Omit clean LFW images from training (use only cloaked). "
             "Avoids polluting the real class with face crops similar to GAN images.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        default=False,
        help="L2-normalize extracted LFW/ProGAN features. MUST match the base FF++ "
             "feature convention (which is RAW). Leave off (raw) for a valid probe.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Discover cloaked & clean image pairs
    # ------------------------------------------------------------------
    print("\n[1/6] Discovering cloaked/clean image pairs...")
    cloaked_paths, clean_paths = build_cloaked_manifest(
        args.cloaked_train_dir, args.lfw_manifest
    )
    print(f"  Found {len(cloaked_paths)} cloaked images with clean matches.")
    if len(cloaked_paths) == 0:
        print("ERROR: No cloaked images found. Run PGD first.")
        sys.exit(1)

    # Save manifest CSVs for reference
    cloaked_manifest = pd.DataFrame({
        "image_path": [str(p.relative_to(REPO_ROOT)) for p in cloaked_paths],
        "label": [0] * len(cloaked_paths),
    })
    clean_manifest = pd.DataFrame({
        "image_path": [str(p.relative_to(REPO_ROOT)) for p in clean_paths],
        "label": [0] * len(clean_paths),
    })
    args.cloaked_features_out.parent.mkdir(parents=True, exist_ok=True)
    cloaked_manifest.to_csv(
        args.cloaked_features_out.parent / "cloaked_lfw_train_manifest.csv", index=False
    )
    clean_manifest.to_csv(
        args.cloaked_features_out.parent / "clean_lfw_train_manifest.csv", index=False
    )
    print(f"  Manifests saved to {args.cloaked_features_out.parent}")

    # ------------------------------------------------------------------
    # 1b. Discover ProGAN training fakes (if requested)
    # ------------------------------------------------------------------
    progan_train_paths = []
    if args.n_progan_train > 0:
        print(f"\n[1b] Finding {args.n_progan_train} non-test ProGAN fake images...")
        progan_train_paths = find_progan_train_fakes(
            args.progan_dir, args.progan_test_manifest, args.n_progan_train, args.seed
        )
        print(f"  Found {len(progan_train_paths)} ProGAN training fakes.")
        progan_train_manifest = pd.DataFrame({
            "image_path": [str(p.relative_to(REPO_ROOT)) for p in progan_train_paths],
            "label": [1] * len(progan_train_paths),
        })
        progan_train_manifest.to_csv(
            args.progan_train_features_out.parent / "progan_train_manifest.csv", index=False
        )

    # ------------------------------------------------------------------
    # 2. Load CLIP backbone
    # ------------------------------------------------------------------
    print("\n[2/6] Loading CLIP ViT-L/14 backbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    clip_model, preprocess = load_clip(device)
    print("  CLIP loaded.")

    # ------------------------------------------------------------------
    # 3. Extract features for cloaked LFW training images
    # ------------------------------------------------------------------
    # Check if already extracted
    if args.cloaked_features_out.exists():
        print(f"\n[3/6] Loading cached cloaked features from {args.cloaked_features_out}")
        cloaked_data = np.load(args.cloaked_features_out, allow_pickle=True)
        cloaked_feats, cloaked_labels = cloaked_data["features"], cloaked_data["labels"]
    else:
        print("\n[3/6] Extracting CLIP features for cloaked LFW training images...")
        cloaked_feats, cloaked_labels, cloaked_fpaths = extract_features_from_paths(
            cloaked_paths,
            [0] * len(cloaked_paths),
            preprocess,
            clip_model,
            device,
            args.batch_size,
            desc="Cloaked LFW",
            normalize=args.normalize_features,
        )
        np.savez(args.cloaked_features_out,
                 features=cloaked_feats, labels=cloaked_labels, paths=cloaked_fpaths)
    print(f"  Cloaked LFW: {len(cloaked_feats)} features")

    # ------------------------------------------------------------------
    # 4. Extract features for clean LFW training images (optional)
    # ------------------------------------------------------------------
    if args.skip_clean:
        print("\n[4/6] Skipping clean LFW features (--skip_clean set).")
        clean_feats = np.empty((0, cloaked_feats.shape[1]), dtype=np.float32)
        clean_labels = np.empty((0,), dtype=np.int32)
    elif args.clean_features_out.exists():
        print(f"\n[4/6] Loading cached clean features from {args.clean_features_out}")
        clean_data = np.load(args.clean_features_out, allow_pickle=True)
        clean_feats, clean_labels = clean_data["features"], clean_data["labels"]
        print(f"  Clean LFW: {len(clean_feats)} features")
    else:
        print("\n[4/6] Extracting CLIP features for clean LFW training images...")
        clean_feats, clean_labels, clean_fpaths = extract_features_from_paths(
            clean_paths,
            [0] * len(clean_paths),
            preprocess,
            clip_model,
            device,
            args.batch_size,
            desc="Clean LFW",
            normalize=args.normalize_features,
        )
        np.savez(args.clean_features_out,
                 features=clean_feats, labels=clean_labels, paths=clean_fpaths)
        print(f"  Saved {len(clean_feats)} clean features → {args.clean_features_out}")

    # ------------------------------------------------------------------
    # 4b. Extract features for ProGAN training fakes (optional)
    # ------------------------------------------------------------------
    if args.n_progan_train > 0 and progan_train_paths:
        print(f"\n[4b] Extracting CLIP features for {len(progan_train_paths)} ProGAN train fakes...")
        pg_feats, pg_labels, pg_fpaths = extract_features_from_paths(
            progan_train_paths,
            [1] * len(progan_train_paths),
            preprocess,
            clip_model,
            device,
            args.batch_size,
            desc="ProGAN train",
            normalize=args.normalize_features,
        )
        np.savez(args.progan_train_features_out,
                 features=pg_feats, labels=pg_labels, paths=pg_fpaths)
        print(f"  Saved {len(pg_feats)} ProGAN train features → {args.progan_train_features_out}")
    else:
        pg_feats = np.empty((0, cloaked_feats.shape[1]), dtype=np.float32)
        pg_labels = np.empty((0,), dtype=np.int32)
        if args.n_progan_train == 0:
            print("\n[4b] ProGAN training fakes disabled (--n_progan_train 0).")

    del clip_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Load 10k FF++ base features and combine
    # ------------------------------------------------------------------
    print("\n[5/6] Loading base 10k features and combining...")
    base = np.load(args.base_features, allow_pickle=True)
    X_base, y_base = base["features"], base["labels"]
    print(f"  FF++ 10k:       {len(X_base)} ({(y_base == 0).sum()} real, {(y_base == 1).sum()} fake)")
    print(f"  Cloaked LFW:    {len(cloaked_feats)} (all real)")
    print(f"  Clean LFW:      {len(clean_feats)} (all real)")
    print(f"  ProGAN train:   {len(pg_feats)} (all fake)")

    X = np.vstack([X_base, cloaked_feats, clean_feats, pg_feats])
    y = np.concatenate([y_base, cloaked_labels, clean_labels, pg_labels])
    print(f"  Combined: {len(X)} total ({(y == 0).sum()} real, {(y == 1).sum()} fake)")

    # ------------------------------------------------------------------
    # 6. Train probe
    # ------------------------------------------------------------------
    if args.use_lr:
        print(f"\n[6/6] Training LogisticRegression (C={args.lr_C})...")
        clf = LogisticRegression(
            C=args.lr_C,
            max_iter=5000,
            random_state=args.seed,
        )
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        print(f"Train accuracy: {train_acc:.4f}")
    else:
        print("\n[6/6] Training MLPClassifier...")
        hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            max_iter=args.max_iter,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=True,
        )
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        print(f"Train accuracy: {train_acc:.4f}  (n_iter={clf.n_iter_})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.output)
    print(f"\nProbe saved → {args.output}")


if __name__ == "__main__":
    main()
