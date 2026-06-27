import argparse
import io
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def jpeg_compress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def downsample_upsample(img):
    w, h = img.size
    small_w, small_h = max(1, w // 2), max(1, h // 2)
    small = img.resize((small_w, small_h), Image.LANCZOS)
    return small.resize((w, h), Image.BILINEAR)


def main():
    parser = argparse.ArgumentParser(
        description="Apply robustness transforms to cloaked images"
    )
    parser.add_argument("--cloaked_dir", type=Path, required=True,
                        help="Directory containing cloaked PNGs")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Root output directory; subdirs created per transform")
    parser.add_argument("--recognizer", choices=["arcface", "facenet"], default="arcface",
                        help="Which cloaked images to process. arcface: *_cloaked.png (not facenet). "
                             "facenet: *_facenet_cloaked.png. Output subdirs get a _{recognizer} suffix.")
    args = parser.parse_args()

    cloaked_dir = args.cloaked_dir
    if not cloaked_dir.exists():
        raise FileNotFoundError(f"cloaked_dir not found: {cloaked_dir}")

    all_files = sorted(cloaked_dir.glob("*.png"))
    if args.recognizer == "arcface":
        cloaked_files = [f for f in all_files if f.name.endswith("_cloaked.png") and "facenet" not in f.name]
        subdir_suffix = ""
    else:
        cloaked_files = [f for f in all_files if f.name.endswith("_facenet_cloaked.png")]
        subdir_suffix = "_facenet"

    print(f"Found {len(cloaked_files)} {args.recognizer} cloaked images in {cloaked_dir}")

    transforms_config = [
        ("jpeg75",     lambda img: jpeg_compress(img, 75)),
        ("jpeg50",     lambda img: jpeg_compress(img, 50)),
        ("downsample", downsample_upsample),
    ]

    for subdir_name, transform_fn in transforms_config:
        out_subdir = args.output_dir / f"{subdir_name}{subdir_suffix}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(cloaked_files, desc=f"  {subdir_name}{subdir_suffix}"):
            img = Image.open(src).convert("RGB")
            transformed = transform_fn(img)
            transformed.save(out_subdir / src.name)
        count = len(list(out_subdir.glob("*.png")))
        print(f"  {subdir_name}{subdir_suffix}: saved {count} images → {out_subdir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
