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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def infer_path_base(manifest_path, rows):
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
        "Cannot resolve manifest paths — check data files exist and --manifest is correct."
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


def build_cloaked_index(cloaked_dir):
    """Return dict: original_stem -> Path for all files directly in cloaked_dir."""
    index = {}
    if not cloaked_dir.exists():
        return index
    for p in cloaked_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        # Strip known cloaking suffixes to recover original stem
        for suffix in ('_high_cloaked', '_low_cloaked', '_mid_cloaked', '_cloaked', '_pgd', '_adv'):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        index[stem] = p
    return index


def find_cloaked_image(cloaked_index, original_raw):
    orig_stem = Path(original_raw.replace('\\', '/')).stem
    return cloaked_index.get(orig_stem)


def compute_ssim_psnr(orig_path, cloaked_path):
    orig = np.array(Image.open(orig_path).convert('RGB'))
    cloaked = np.array(Image.open(cloaked_path).convert('RGB'))
    if orig.shape != cloaked.shape:
        cloaked = np.array(
            Image.fromarray(cloaked).resize((orig.shape[1], orig.shape[0]), Image.LANCZOS)
        )
    orig_f = orig.astype(np.float64) / 255.0
    cloaked_f = cloaked.astype(np.float64) / 255.0
    ssim_val = structural_similarity(orig_f, cloaked_f, channel_axis=2, data_range=1.0)
    psnr_val = peak_signal_noise_ratio(orig_f, cloaked_f, data_range=1.0)
    return float(ssim_val), float(psnr_val)


def extract_arcface_embeddings(pairs, size, ctx_id):
    """pairs: list of (key, Path). Returns dict key -> embedding."""
    app = FaceAnalysis(
        name='buffalo_l',
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
    for key, path in tqdm(pairs, desc='ArcFace embeddings'):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size))
        feat = rec.get_feat(img).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        embeddings[key] = feat

    del app
    torch.cuda.empty_cache()
    return embeddings


def extract_facenet_embeddings(pairs, size, device):
    """pairs: list of (key, Path). Returns dict key -> embedding."""
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    embeddings = {}
    with torch.no_grad():
        for key, path in tqdm(pairs, desc='FaceNet embeddings'):
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat /= norm
            embeddings[key] = feat

    del model
    torch.cuda.empty_cache()
    return embeddings


def evaluate_cloaked_loo(groups, gallery_embs, query_embs, path_key, orig_to_cloaked):
    """
    gallery_embs  : dict  original-path-key  -> embedding
    query_embs    : dict  cloaked-path-string -> embedding
    orig_to_cloaked: dict original-path-key  -> cloaked-path-string
                     Pass {key: key} to run on original images (sanity-check mode).

    LOO: for each query, exclude corresponding original from gallery mean.
    Returns list of result dicts with own_sim and best_other_sim for diagnostics.
    """
    identities = list(groups.keys())

    identity_sums = {}
    identity_counts = {}
    for iid, rows in groups.items():
        vecs = [gallery_embs[r[path_key]] for r in rows if r[path_key] in gallery_embs]
        if vecs:
            identity_sums[iid] = np.stack(vecs).sum(axis=0)
            identity_counts[iid] = len(vecs)

    identity_means_norm = {}
    for iid in identities:
        if iid not in identity_sums:
            continue
        m = identity_sums[iid] / identity_counts[iid]
        n = np.linalg.norm(m)
        if n > 0:
            m /= n
        identity_means_norm[iid] = m

    valid_ids = set(identity_means_norm.keys())

    results = []
    for iid, rows in tqdm(groups.items(), desc='LOO eval'):
        if iid not in identity_sums:
            continue
        sum_i = identity_sums[iid]

        for row in rows:
            orig_key = row[path_key]
            cloaked_key = orig_to_cloaked.get(orig_key)
            if cloaked_key is None or cloaked_key not in query_embs:
                continue

            query_emb = query_embs[cloaked_key]
            orig_emb = gallery_embs.get(orig_key)

            # LOO: subtract original image from gallery sum before computing mean
            loo_sum = (sum_i - orig_emb) if orig_emb is not None else sum_i.copy()
            loo_norm = np.linalg.norm(loo_sum)
            if loo_norm > 0:
                loo_sum = loo_sum / loo_norm
            own_sim = float(np.dot(query_emb, loo_sum))

            best_id = iid
            best_sim = own_sim
            for other_id in valid_ids:
                if other_id == iid:
                    continue
                sim = float(np.dot(query_emb, identity_means_norm[other_id]))
                if sim > best_sim:
                    best_sim = sim
                    best_id = other_id

            results.append({
                'identity_id': iid,
                'orig_key': orig_key,
                'cloaked_path': cloaked_key,
                'correct': 1 if best_id == iid else 0,
                'own_sim': own_sim,
                'best_other_sim': best_sim if best_id != iid else own_sim,
            })

    return results


def print_sim_diagnostics(results, label=''):
    own_sims = [r['own_sim'] for r in results]
    best_sims = [r['best_other_sim'] for r in results]
    correct = sum(r['correct'] for r in results)
    print(f"\n[DIAG{' ' + label if label else ''}]")
    print(f"  n={len(results)}, correct={correct}, acc={correct/len(results):.4f}")
    print(f"  own_sim  : mean={np.mean(own_sims):.4f}  std={np.std(own_sims):.4f}"
          f"  min={np.min(own_sims):.4f}  max={np.max(own_sims):.4f}")
    print(f"  best_sim : mean={np.mean(best_sims):.4f}  std={np.std(best_sims):.4f}"
          f"  min={np.min(best_sims):.4f}  max={np.max(best_sims):.4f}")
    print(f"  margin (own-best): mean={np.mean(np.array(own_sims)-np.array(best_sims)):.4f}")
    print(f"  Sample (first 5):")
    for r in results[:5]:
        print(f"    {r['identity_id']:30s}  own={r['own_sim']:+.4f}  best={r['best_other_sim']:+.4f}"
              f"  {'OK' if r['correct'] else 'FAIL'}")


def gallery_intra_similarity(groups, gallery_embs, path_key):
    """Mean pairwise cosine similarity within each identity. Should be high (>0.4) for a working model."""
    all_sims = []
    for iid, rows in groups.items():
        vecs = [gallery_embs[r[path_key]] for r in rows if r[path_key] in gallery_embs]
        if len(vecs) < 2:
            continue
        mat = np.stack(vecs)
        # upper triangle of cosine similarity matrix (already L2-normalised)
        n = len(vecs)
        for i in range(n):
            for j in range(i + 1, n):
                all_sims.append(float(np.dot(mat[i], mat[j])))
    return all_sims


def main():
    parser = argparse.ArgumentParser(description='Evaluate cloaking ASR, SSIM, PSNR on LFW')
    parser.add_argument('--manifest', type=Path, default=Path('data/lfw/manifest.csv'))
    parser.add_argument('--method', required=True,
                        help='Cloaking method name, e.g. fawkes')
    parser.add_argument('--cloaked-dir', type=Path, required=True,
                        help='Directory with per-identity subfolders of cloaked images')
    parser.add_argument('--recognizer', choices=['arcface', 'facenet'], required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/privacy'))
    parser.add_argument('--arcface-size', type=int, default=112)
    parser.add_argument('--facenet-size', type=int, default=160)
    parser.add_argument('--sanity-check', action='store_true',
                        help='Before cloaked eval, run LOO on original images. '
                             'Should reproduce baseline accuracy. Flags eval-logic bugs.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx_id = 0 if device == 'cuda' else -1
    print(f"Device: {device}")

    manifest_path = args.manifest.resolve()
    rows = load_manifest(manifest_path)
    print(f"Loaded {len(rows)} manifest rows")

    groups = group_by_identity(rows)
    print(f"Identities: {len(groups)}")

    base = infer_path_base(manifest_path, rows)
    print(f"Path base: {base}")

    path_key = 'arcface_path' if args.recognizer == 'arcface' else 'facenet_path'
    img_size = args.arcface_size if args.recognizer == 'arcface' else args.facenet_size

    # Build flat index of all cloaked images in cloaked_dir
    cloaked_index = build_cloaked_index(args.cloaked_dir)
    print(f"Cloaked images indexed: {len(cloaked_index)}")

    # Match each manifest row to its cloaked counterpart; compute SSIM/PSNR
    orig_to_cloaked = {}
    ssim_psnr_map = {}
    missing = 0
    for row in rows:
        orig_raw = row[path_key]
        orig_path = resolve(orig_raw, base)
        cloaked_path = find_cloaked_image(cloaked_index, orig_raw)
        if cloaked_path is None:
            missing += 1
            continue
        orig_to_cloaked[orig_raw] = str(cloaked_path)
        try:
            ssim_val, psnr_val = compute_ssim_psnr(orig_path, Path(orig_to_cloaked[orig_raw]))
        except Exception as e:
            print(f"SSIM/PSNR error for {cloaked_path}: {e}")
            ssim_val, psnr_val = float('nan'), float('nan')
        ssim_psnr_map[orig_raw] = (ssim_val, psnr_val)

    print(f"Cloaked images matched: {len(orig_to_cloaked)}, missing: {missing}")
    if not orig_to_cloaked:
        raise RuntimeError(
            f"No cloaked images found in {args.cloaked_dir}. "
            "Check --cloaked-dir points to the directory with identity subfolders."
        )

    # Gallery: original image embeddings (all rows, deduped by key)
    seen = set()
    gallery_pairs = []
    for row in rows:
        k = row[path_key]
        if k not in seen:
            seen.add(k)
            gallery_pairs.append((k, resolve(k, base)))

    print(f"Extracting gallery embeddings ({args.recognizer}, {len(gallery_pairs)} images)...")
    if args.recognizer == 'arcface':
        gallery_embs = extract_arcface_embeddings(gallery_pairs, img_size, ctx_id)
    else:
        gallery_embs = extract_facenet_embeddings(gallery_pairs, img_size, device)

    # Sanity-check 1: intra-identity gallery similarity
    intra_sims = gallery_intra_similarity(groups, gallery_embs, path_key)
    print(f"\n[DIAG gallery intra-identity cosine similarity]")
    print(f"  mean={np.mean(intra_sims):.4f}  std={np.std(intra_sims):.4f}"
          f"  min={np.min(intra_sims):.4f}  max={np.max(intra_sims):.4f}")
    print(f"  (Expected >0.4 for a working recogniser; ~0 means garbage embeddings)")

    # Sanity-check 2: LOO on originals should reproduce baseline accuracy
    if args.sanity_check:
        print("\n[SANITY CHECK] Running LOO on original (un-cloaked) images...")
        orig_identity_map = {row[path_key]: row[path_key] for row in rows if row[path_key] in gallery_embs}
        sanity_results = evaluate_cloaked_loo(groups, gallery_embs, gallery_embs, path_key, orig_identity_map)
        print_sim_diagnostics(sanity_results, label='SANITY original images')
        print("  (Accuracy should match baseline_recognition.csv; if not, eval logic has a bug)")

    # Query: cloaked image embeddings (deduped by cloaked path)
    seen_c = set()
    query_pairs = []
    for cloaked_key in orig_to_cloaked.values():
        if cloaked_key not in seen_c:
            seen_c.add(cloaked_key)
            query_pairs.append((cloaked_key, Path(cloaked_key)))

    print(f"Extracting query embeddings ({args.recognizer}, {len(query_pairs)} cloaked images)...")
    if args.recognizer == 'arcface':
        query_embs = extract_arcface_embeddings(query_pairs, img_size, ctx_id)
    else:
        query_embs = extract_facenet_embeddings(query_pairs, img_size, device)

    # LOO evaluation
    loo_results = evaluate_cloaked_loo(groups, gallery_embs, query_embs, path_key, orig_to_cloaked)
    print_sim_diagnostics(loo_results, label='CLOAKED')

    # Assemble output rows
    out_rows = []
    for r in loo_results:
        ssim_val, psnr_val = ssim_psnr_map.get(r['orig_key'], (float('nan'), float('nan')))
        out_rows.append({
            'method': args.method,
            'recognizer': args.recognizer,
            'identity_id': r['identity_id'],
            'image_path': r['cloaked_path'],
            'correct': r['correct'],
            'ssim': f"{ssim_val:.6f}",
            'psnr': f"{psnr_val:.4f}",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.method}_{args.recognizer}_metrics.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['method', 'recognizer', 'identity_id', 'image_path', 'correct', 'ssim', 'psnr']
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved {len(out_rows)} rows to {out_path}")

    # Summary table
    n_total = len(out_rows)
    n_correct = sum(r['correct'] for r in out_rows)
    asr = 1.0 - (n_correct / n_total) if n_total > 0 else float('nan')
    ssim_vals = [float(r['ssim']) for r in out_rows if r['ssim'] not in ('nan', 'inf')]
    psnr_vals = [float(r['psnr']) for r in out_rows if r['psnr'] not in ('nan', 'inf')]
    mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else float('nan')
    mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else float('nan')
    mean_own = float(np.mean([r['own_sim'] for r in loo_results])) if loo_results else float('nan')
    mean_best = float(np.mean([r['best_other_sim'] for r in loo_results])) if loo_results else float('nan')

    print(f"\n{'=' * 52}")
    print(f"  Method         : {args.method}")
    print(f"  Recognizer     : {args.recognizer}")
    print(f"  Images         : {n_total}")
    print(f"  ASR            : {asr:.4f}  ({n_total - n_correct}/{n_total} fooled)")
    print(f"  Mean SSIM      : {mean_ssim:.4f}")
    print(f"  Mean PSNR      : {mean_psnr:.2f} dB")
    print(f"  Mean own_sim   : {mean_own:.4f}  (cloaked vs true-id gallery)")
    print(f"  Mean best_sim  : {mean_best:.4f}  (cloaked vs best-other-id gallery)")
    print(f"{'=' * 52}")


if __name__ == '__main__':
    main()
