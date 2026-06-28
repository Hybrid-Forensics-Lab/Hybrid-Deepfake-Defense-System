import argparse
import pandas as pd
from pathlib import Path


FAKE_TYPES = ["deepfakes", "face2face", "faceswap", "neuraltextures"]


def build_manifest(data_root: Path, output: Path, seed: int) -> pd.DataFrame:
    rows = []

    real_dir = data_root / "256x256" / "real"
    for img_path in sorted(real_dir.glob("*")):
        if img_path.is_file():
            rows.append({
                "image_path": str(img_path),
                "label": 0,
                "forgery_type": "real",
                "resolution": 256,
            })

    for forgery_type in FAKE_TYPES:
        fake_dir = data_root / "256x256" / "fake" / forgery_type
        for img_path in sorted(fake_dir.glob("*")):
            if img_path.is_file():
                rows.append({
                    "image_path": str(img_path),
                    "label": 1,
                    "forgery_type": forgery_type,
                    "resolution": 256,
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame):
    total = len(df)
    real_count = (df["label"] == 0).sum()
    fake_counts = df[df["label"] == 1].groupby("forgery_type").size()

    print(f"Total images : {total}")
    print(f"Real         : {real_count}")
    for ft in FAKE_TYPES:
        count = fake_counts.get(ft, 0)
        print(f"  {ft:<20}: {count}")
    status = "PASS" if total == 1000 else "FAIL"
    print(f"Total == 1000: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FF++ manifest CSV")
    parser.add_argument("--data_root", type=Path, default=Path("data/ff_plus_plus"))
    parser.add_argument("--output", type=Path, default=Path("data/ff_plus_plus/manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_manifest(args.data_root, args.output, args.seed)
    print_summary(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
