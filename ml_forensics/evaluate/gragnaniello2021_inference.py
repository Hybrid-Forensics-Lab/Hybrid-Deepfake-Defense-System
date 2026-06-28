import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Manifest CSV with image_path column.")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file.")
    parser.add_argument("--output-csv", required=True, help="Output raw CSV path.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    model_dir = repo / "ml_forensics" / "models" / "GANimageDetection"
    sys.path.insert(0, str(model_dir))
    from resnet50nodown import resnet50nodown

    from torch.cuda import is_available as cuda_available
    device = "cuda:0" if cuda_available() else "cpu"
    print(f"Device: {device}")

    weights_path = Path(args.weights)
    print(f"Loading weights: {weights_path}")
    net = resnet50nodown(device, str(weights_path))

    manifest = pd.read_csv(args.manifest)
    print(f"Images to score: {len(manifest)}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fid:
        fid.write("filename,logit,time\n")
        for img_path in tqdm(manifest["image_path"], desc="Scoring"):
            tic = time.time()
            img = Image.open(img_path).convert("RGB")
            img.load()
            logit = net.apply(img)
            toc = time.time()
            fid.write(f"{img_path},{logit},{toc - tic}\n")
            fid.flush()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
