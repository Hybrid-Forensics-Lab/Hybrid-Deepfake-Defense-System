import argparse
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Run Fawkes cloaking on LFW images")
    p.add_argument("--recognizer", required=True, choices=["arcface", "facenet"])
    p.add_argument("--manifest", default="data/lfw/manifest.csv")
    p.add_argument("--lfw-root", default="data/lfw")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fawkes-script", default="ml_privacy/models/fawkes/fawkes/protection.py")
    p.add_argument("--mode", default="mid")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def get_identity_folders(manifest_path, lfw_root, recognizer):
    df = pd.read_csv(manifest_path)
    identities = df["identity_id"].dropna().unique()
    subdir = "arcface_112" if recognizer == "arcface" else "facenet_160"
    return sorted(lfw_root / subdir / identity for identity in identities)


def already_done(output_dir, identity_name, min_files=5):
    if not output_dir.exists():
        return False
    matched = list(output_dir.glob(f"{identity_name}*_cloaked.png"))
    return len(matched) >= min_files


def run_fawkes(fawkes_script, identity_folder, mode, gpu, batch_size):
    cmd = [
        "python3", str(fawkes_script.resolve()),
        "-d", str(identity_folder.resolve()),
        "--mode", mode,
        "--gpu", gpu,
        "--format", "png",
        "--no-align",
        "--debug",
        "--batch-size", str(batch_size),
    ]
    result = subprocess.run(cmd, cwd=str(fawkes_script.parent.resolve()))
    return result.returncode


def move_cloaked(identity_folder, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cloaked = list(identity_folder.glob("*_cloaked.png"))
    for f in cloaked:
        shutil.move(str(f), str(output_dir / f.name))
    return len(cloaked)


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    lfw_root = Path(args.lfw_root)
    fawkes_script = Path(args.fawkes_script)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/privacy/fawkes") / args.recognizer
    )

    identity_folders = get_identity_folders(manifest_path, lfw_root, args.recognizer)
    total = len(identity_folders)

    processed = 0
    skipped = 0
    total_moved = 0

    for i, folder in enumerate(identity_folders, start=1):
        identity_name = folder.name

        if already_done(output_dir, identity_name):
            print(f"[{i}/{total}] Skipping {identity_name} — already done")
            skipped += 1
            continue

        print(f"[{i}/{total}] Processing identity: {identity_name}")

        rc = run_fawkes(fawkes_script, folder, args.mode, args.gpu, args.batch_size)
        if rc != 0:
            print(f"  WARNING: Fawkes exited with code {rc} for {identity_name}", file=sys.stderr)

        moved = move_cloaked(folder, output_dir)
        print(f"  Moved {moved} cloaked file(s) to {output_dir}")
        total_moved += moved
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total images moved: {total_moved}")


if __name__ == "__main__":
    main()
