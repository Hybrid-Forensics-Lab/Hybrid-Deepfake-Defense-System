import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def infer_path_base(manifest_path, rows):
    """Find the filesystem ancestor that makes manifest paths resolvable.

    Paths in the manifest use Windows backslashes and are relative to the
    repo parent (e.g. 'Hybrid-Deepfake-Defense-System\\data\\lfw\\...')
    rather than the repo root itself.
    """
    for row in rows:
        raw = row.get('arcface_path') or row.get('facenet_path')
        if not raw:
            continue
        converted = raw.replace('\\', '/')
        p = Path(converted)
        if p.is_absolute():
            return Path('/')
        for ancestor in manifest_path.resolve().parents:
            if (ancestor / p).exists():
                return ancestor
        break
    raise RuntimeError(
        "Cannot resolve manifest paths — check that data files are present "
        "and that --manifest points to the correct location."
    )


def resolve(raw, base):
    return base / Path(raw.replace('\\', '/'))


def group_by_identity(rows):
    groups = {}
    for row in rows:
        iid = row['identity_id']
        if iid not in groups:
            groups[iid] = []
        groups[iid].append(row)
    return groups


def extract_arcface_embeddings(rows, arcface_size, base, ctx_id):
    app = FaceAnalysis(
        name='buffalo_l',
        # allowed_modules=['recognition'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    app.prepare(ctx_id=ctx_id)

    rec = app.models.get('recognition')
    if rec is None:
        for m in app.models.values():
            if hasattr(m, 'get_feat'):
                rec = m
                break
    if rec is None:
        raise RuntimeError("buffalo_l recognition model not found")

    embeddings = {}
    for row in tqdm(rows, desc='ArcFace embeddings'):
        img_path = resolve(row['arcface_path'], base)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        if img.shape[:2] != (arcface_size, arcface_size):
            img = cv2.resize(img, (arcface_size, arcface_size))
        feat = rec.get_feat(img).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        embeddings[row['arcface_path']] = feat

    del app
    torch.cuda.empty_cache()
    return embeddings


def extract_facenet_embeddings(rows, facenet_size, base, device):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((facenet_size, facenet_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    embeddings = {}
    with torch.no_grad():
        for row in tqdm(rows, desc='FaceNet embeddings'):
            img_path = resolve(row['facenet_path'], base)
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm
            embeddings[row['facenet_path']] = feat

    del model
    torch.cuda.empty_cache()
    return embeddings


def evaluate_loo(groups, embeddings, path_key):
    """Leave-one-out top-1 accuracy across all identities.

    For each image of identity i:
      - own_mean  = mean of the other 4 embeddings of identity i  (LOO)
      - other_mean_k = mean of all 5 embeddings for every other identity k
      - correct   = own cosine-sim is strictly the highest among all 200 means
    """
    identities = list(groups.keys())

    # Precompute per-identity embedding sums for efficient LOO means
    identity_sums = {}
    identity_counts = {}
    for iid, rows in groups.items():
        vecs = np.stack([embeddings[r[path_key]] for r in rows])
        identity_sums[iid] = vecs.sum(axis=0)
        identity_counts[iid] = len(rows)

    # Precompute normalized full means for all identities once
    identity_means_norm = {}
    for iid in identities:
        m = identity_sums[iid] / identity_counts[iid]
        n = np.linalg.norm(m)
        if n > 0:
            m /= n
        identity_means_norm[iid] = m

    results = []
    for iid, rows in tqdm(groups.items(), desc='LOO evaluation'):
        sum_i = identity_sums[iid]

        for row in rows:
            emb = embeddings[row[path_key]]

            # LOO direction for own identity: sum of the other images (normalized for cosine sim)
            own_mean = sum_i - emb
            own_norm = np.linalg.norm(own_mean)
            if own_norm > 0:
                own_mean = own_mean / own_norm
            own_sim = float(np.dot(emb, own_mean))

            best_id = iid
            best_sim = own_sim

            for other_id in identities:
                if other_id == iid:
                    continue
                sim = float(np.dot(emb, identity_means_norm[other_id]))
                if sim > best_sim:
                    best_sim = sim
                    best_id = other_id

            results.append({
                'identity_id': iid,
                'image_path': row[path_key],
                'correct': 1 if best_id == iid else 0,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline face recognition evaluation on LFW')
    parser.add_argument('--manifest', type=Path, default=Path('data/lfw/manifest.csv'))
    parser.add_argument('--arcface-size', type=int, default=112)
    parser.add_argument('--facenet-size', type=int, default=160)
    parser.add_argument('--output', type=Path, default=Path('results/privacy/baseline_recognition.csv'))
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx_id = 0 if device == 'cuda' else -1
    print(f"Device: {device}")

    manifest_path = args.manifest.resolve()
    rows = load_manifest(manifest_path)
    print(f"Loaded {len(rows)} rows")

    groups = group_by_identity(rows)
    print(f"Identities: {len(groups)}, images per identity: {[len(v) for v in groups.values()][:5]} ...")

    base = infer_path_base(manifest_path, rows)
    print(f"Path base: {base}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    recognizers = [
        ('arcface', 'arcface_path', lambda: extract_arcface_embeddings(rows, args.arcface_size, base, ctx_id)),
        ('facenet', 'facenet_path', lambda: extract_facenet_embeddings(rows, args.facenet_size, base, device)),
    ]

    for recognizer, path_key, extract_fn in recognizers:
        print(f"\n=== {recognizer} ===")
        embeddings = extract_fn()
        loo_results = evaluate_loo(groups, embeddings, path_key)
        n_correct = sum(r['correct'] for r in loo_results)
        acc = n_correct / len(loo_results)
        print(f"{recognizer} top-1 accuracy: {acc:.4f}  ({n_correct}/{len(loo_results)})")
        for r in loo_results:
            all_results.append({
                'recognizer': recognizer,
                'identity_id': r['identity_id'],
                'image_path': r['image_path'],
                'correct': r['correct'],
            })

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['recognizer', 'identity_id', 'image_path', 'correct'])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved {len(all_results)} rows to {args.output}")


if __name__ == '__main__':
    main()
