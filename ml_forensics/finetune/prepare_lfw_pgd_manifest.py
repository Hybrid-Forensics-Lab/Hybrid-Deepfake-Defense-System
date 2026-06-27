"""
Prepare a filtered LFW manifest for adversarial-aware PGD cloaking.

Reads the test-set cloaked_images directory to find excluded identities,
then writes a disjoint subset of the LFW manifest for use as training cloaks.
"""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_excluded_identities(cloaked_test_dir):
    excluded = set()
    for f in Path(cloaked_test_dir).glob("*_cloaked.png"):
        # Strip _cloaked and _facenet suffixes, then strip trailing _NNNN
        stem = f.stem.replace("_facenet_cloaked", "").replace("_cloaked", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            excluded.add(parts[0])
    return excluded


def main():
    parser = argparse.ArgumentParser(
        description="Create disjoint LFW manifest for adversarial-aware PGD training."
    )
    parser.add_argument(
        "--lfw_manifest",
        type=Path,
        default=REPO_ROOT / "data/lfw/manifest.csv",
        help="Path to data/lfw/manifest.csv",
    )
    parser.add_argument(
        "--cloaked_test_dir",
        type=Path,
        default=REPO_ROOT / "results/phase2/privacy/cloaked_images",
        help="Directory containing test-set cloaked images (excluded identities).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/phase2/privacy/train_pgd_run/lfw_pgd_manifest.csv",
        help="Output path for filtered manifest.",
    )
    parser.add_argument(
        "--max_per_identity",
        type=int,
        default=5,
        help="Max images per identity (default: all = 5).",
    )
    args = parser.parse_args()

    excluded = get_excluded_identities(args.cloaked_test_dir)
    print(f"Excluded identities (test set): {len(excluded)}")

    manifest = pd.read_csv(args.lfw_manifest)
    print(f"Total LFW manifest rows: {len(manifest)}")

    disjoint = manifest[~manifest["identity_id"].isin(excluded)].copy()
    print(
        f"Disjoint rows: {len(disjoint)} from "
        f"{disjoint['identity_id'].nunique()} identities"
    )

    if args.max_per_identity:
        disjoint = (
            disjoint.groupby("identity_id", group_keys=False)
            .apply(lambda g: g.head(args.max_per_identity))
            .reset_index(drop=True)
        )
        print(f"After capping at {args.max_per_identity} per identity: {len(disjoint)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    disjoint.to_csv(args.output, index=False)
    print(f"Saved filtered manifest → {args.output}")


if __name__ == "__main__":
    main()
