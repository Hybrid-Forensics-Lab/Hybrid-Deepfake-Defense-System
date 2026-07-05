"""Recompute the forensic conflict numbers with the DEPLOYED corrected probe.

Background: the original conflict artifacts in results/phase2/conflict/ were
contaminated. The `ft_univfd_adv_thresh*` CSVs were produced from an eval cloaked
feature cache that had been L2-normalized while the probe expects RAW features
(the normalization artifact, phase2.md Section 8), which collapsed cloaked FPR to
0%. The `selected_operating_point.txt` and `conflict_summary.csv` described an
even older MLP-10k-plus-Wang2020-gate plan that was later dropped.

This script regenerates the conflict numbers from scratch using the deployed
probe (mlp_probe_adv_aware_raw.pkl, RAW features) on:
  - FF++ test set features (raw, cached)
  - ProGAN test set features (raw, cached)
  - the 400 eval cloaked PNGs, RE-EXTRACTED as raw here (overwriting the
    contaminated clip_features_eval_cloaked.npz)

It VALIDATES the recomputed headline numbers against the phase2.md values and
aborts before writing if anything is off. Only on a clean validation does it
overwrite the canonical conflict artifacts.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from ml_forensics.finetune.train_adv_aware_probe import load_clip, extract_features_from_paths

import torch

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# phase2.md deployed-probe reference values (Sections 5/8/9) for validation.
REF = {
    "ff_auc": 0.8116,
    "progan_auc": 0.8294,
    "ff_recall@0.3": 0.620,
    "cloaked_fpr@0.3": 0.1775,
    "cloaked_fpr@0.4": 0.1225,
}


def predict_prob(probe, X):
    return probe.predict_proba(X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path,
                    default=REPO_ROOT / "results/phase2/forensics/mlp_probe_adv_aware_raw.pkl")
    ap.add_argument("--fftest_features", type=Path,
                    default=REPO_ROOT / "results/phase2/forensics/clip_features_fftest.npz")
    ap.add_argument("--progan_features", type=Path,
                    default=REPO_ROOT / "results/phase2/forensics/clip_features_progan.npz")
    ap.add_argument("--cloaked_dir", type=Path,
                    default=REPO_ROOT / "results/phase2/privacy/cloaked_images")
    ap.add_argument("--cloaked_features_out", type=Path,
                    default=REPO_ROOT / "results/phase2/forensics/clip_features_eval_cloaked.npz")
    ap.add_argument("--conflict_dir", type=Path,
                    default=REPO_ROOT / "results/phase2/conflict")
    ap.add_argument("--forensics_dir", type=Path,
                    default=REPO_ROOT / "results/phase2/forensics")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tol_rate", type=float, default=0.01)
    ap.add_argument("--tol_auc", type=float, default=0.005)
    args = ap.parse_args()

    probe = joblib.load(args.probe)
    print(f"Loaded probe: {type(probe).__name__} C={getattr(probe, 'C', '?')}  from {args.probe.name}")

    # ---- FF++ test (raw cached features) ----
    ff = np.load(args.fftest_features, allow_pickle=True)
    X_ff, y_ff, ff_paths = ff["features"], ff["labels"].astype(int), ff["paths"]
    p_ff = predict_prob(probe, X_ff)
    ff_auc = roc_auc_score(y_ff, p_ff)

    # ---- ProGAN test (raw cached features) ----
    pg = np.load(args.progan_features, allow_pickle=True)
    X_pg, y_pg, pg_paths = pg["features"], pg["labels"].astype(int), pg["paths"]
    p_pg = predict_prob(probe, X_pg)
    progan_auc = roc_auc_score(y_pg, p_pg)

    # ---- Cloaked eval set: RE-EXTRACT raw features (fixes the contaminated cache) ----
    cloaked_paths = sorted(args.cloaked_dir.glob("*_cloaked.png"))
    if len(cloaked_paths) == 0:
        sys.exit(f"No cloaked PNGs found in {args.cloaked_dir}")
    print(f"Re-extracting RAW CLIP features for {len(cloaked_paths)} cloaked eval images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = load_clip(device)
    feats, _, fpaths = extract_features_from_paths(
        cloaked_paths, [0] * len(cloaked_paths), preprocess, clip_model, device,
        args.batch_size, desc="Cloaked eval (raw)", normalize=False)
    del clip_model
    torch.cuda.empty_cache()
    norms = np.linalg.norm(feats[:5], axis=1)
    print(f"  sanity: first-5 feature L2 norms {norms.round(2)} (must be >>1, i.e. RAW)")
    if norms.mean() < 5:
        sys.exit("ABORT: re-extracted cloaked features look normalized, not raw.")
    np.savez(args.cloaked_features_out, features=feats, paths=fpaths)
    print(f"  overwrote {args.cloaked_features_out.name} with RAW features")

    is_facenet = np.array(["_facenet_cloaked" in Path(p).name for p in fpaths])
    p_cloaked = predict_prob(probe, feats)
    p_arc = p_cloaked[~is_facenet]
    p_fnet = p_cloaked[is_facenet]
    print(f"  cloaked split: {len(p_arc)} ArcFace, {len(p_fnet)} FaceNet")

    # ---- per-threshold metrics ----
    def recall_at(t):
        return float((p_ff[y_ff == 1] > t).mean())

    def ff_real_fpr_at(t):
        return float((p_ff[y_ff == 0] > t).mean())

    def cloaked_fpr_at(scores, t):
        return float((scores > t).mean())

    # ---- VALIDATION against phase2.md ----
    got = {
        "ff_auc": ff_auc,
        "progan_auc": progan_auc,
        "ff_recall@0.3": recall_at(0.3),
        "cloaked_fpr@0.3": cloaked_fpr_at(p_cloaked, 0.3),
        "cloaked_fpr@0.4": cloaked_fpr_at(p_cloaked, 0.4),
    }
    print("\n=== VALIDATION vs phase2.md ===")
    ok = True
    for k, ref in REF.items():
        tol = args.tol_auc if k.endswith("auc") else args.tol_rate
        delta = abs(got[k] - ref)
        status = "PASS" if delta <= tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {k:18s} got={got[k]:.4f}  ref={ref:.4f}  Δ={delta:.4f}  [{status}]")
    if not ok:
        sys.exit("\nABORT: recomputed numbers do not match phase2.md. Nothing written.")
    print("All headline numbers reproduce phase2.md. Writing corrected artifacts.\n")

    args.conflict_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. cloaked_fpr_curve.csv (clean source for figures) ----
    curve = args.conflict_dir / "cloaked_fpr_curve.csv"
    with open(curve, "w") as f:
        f.write("threshold,cloaked_fpr_overall,cloaked_fpr_arcface,cloaked_fpr_facenet,"
                "ff_recall,ff_real_fpr,passes_15pct\n")
        for t in THRESHOLDS:
            f.write(f"{t},{cloaked_fpr_at(p_cloaked,t):.4f},{cloaked_fpr_at(p_arc,t):.4f},"
                    f"{cloaked_fpr_at(p_fnet,t):.4f},{recall_at(t):.4f},{ff_real_fpr_at(t):.4f},"
                    f"{cloaked_fpr_at(p_cloaked,t) < 0.15}\n")
    print(f"wrote {curve.relative_to(REPO_ROOT)}")

    # ---- 2. conflict_summary.csv (deployed adv-aware + preserved comparison rows) ----
    summary = args.conflict_dir / "conflict_summary.csv"
    with open(summary, "w") as f:
        f.write("forensic_model,threshold,num_images,num_flagged,fpr,passes_15pct,ff_recall\n")
        for t in THRESHOLDS:
            nf = int((p_cloaked > t).sum())
            f.write(f"ft_univfd_adv_raw,{t},{len(p_cloaked)},{nf},"
                    f"{cloaked_fpr_at(p_cloaked,t):.4f},{cloaked_fpr_at(p_cloaked,t) < 0.15},"
                    f"{recall_at(t):.4f}\n")
        # Preserved comparison rows (other models / pre-mitigation), from the
        # original conflict matrix; unaffected by the adv-probe normalization bug.
        f.write("wang2020,0.5,400,297,0.7425,False,0.010\n")
        for t, fpr, nfl in [(0.5, 0.6925, 277), (0.6, 0.6525, 261), (0.7, 0.5950, 238),
                            (0.8, 0.5075, 203), (0.9, 0.3900, 156)]:
            f.write(f"univfd_knn,{t},400,{nfl},{fpr},False,\n")
        # Naive MLP-10k probe (pre adversarial-aware), the conflict "before" baseline.
        for t, fpr, nfl, rec in [(0.5, 0.895, 358, 0.664), (0.6, 0.690, 276, 0.652),
                                 (0.7, 0.365, 146, 0.640), (0.8, 0.0975, 39, 0.608),
                                 (0.9, 0.0025, 1, 0.578)]:
            f.write(f"ft_univfd_mlp10k_naive,{t},400,{nfl},{fpr},{fpr < 0.15},{rec}\n")
    print(f"wrote {summary.relative_to(REPO_ROOT)}")

    # ---- 3. per-image cloaked scores ----
    perimg = args.conflict_dir / "ft_univfd_adv_raw_cloaked_scores.csv"
    with open(perimg, "w") as f:
        f.write("image_path,recognizer,synthetic_prob\n")
        for p, s, fn in zip(fpaths, p_cloaked, is_facenet):
            rel = Path(p)
            try:
                rel = rel.relative_to(REPO_ROOT)
            except ValueError:
                pass
            f.write(f"{rel},{'facenet' if fn else 'arcface'},{s:.6f}\n")
    print(f"wrote {perimg.relative_to(REPO_ROOT)}")

    # ---- 4. overwrite FF++ / ProGAN per-image results + metrics with the deployed probe ----
    def write_results(prefix, paths, y, p):
        res = args.forensics_dir / f"{prefix}_results.csv"
        with open(res, "w") as f:
            f.write("image_path,true_label,predicted_prob\n")
            for pa, yi, pi in zip(paths, y, p):
                f.write(f"{pa},{int(yi)},{pi:.6f}\n")
        pred = (p > 0.5).astype(int)
        met = args.forensics_dir / f"{prefix}_metrics.csv"
        with open(met, "w") as f:
            f.write("accuracy,precision,recall,auc_roc\n")
            f.write(f"{accuracy_score(y,pred):.4f},{precision_score(y,pred,zero_division=0):.4f},"
                    f"{recall_score(y,pred,zero_division=0):.4f},{roc_auc_score(y,p):.4f}\n")
        print(f"wrote {res.relative_to(REPO_ROOT)} and {met.name}")

    write_results("ft_univfd_adv_ff", ff_paths, y_ff, p_ff)
    write_results("ft_univfd_adv_progan", pg_paths, y_pg, p_pg)

    # ---- 5. remove the buggy 0%-FPR artifacts ----
    removed = 0
    for f in args.conflict_dir.glob("ft_univfd_adv_thresh*"):
        f.unlink()
        removed += 1
    print(f"removed {removed} buggy ft_univfd_adv_thresh* files")

    # ---- 6. selected_operating_point.txt (corrected) ----
    op = args.conflict_dir / "selected_operating_point.txt"
    fpr03 = cloaked_fpr_at(p_cloaked, 0.3)
    fpr04 = cloaked_fpr_at(p_cloaked, 0.4)
    arc03 = cloaked_fpr_at(p_arc, 0.3)
    with open(op, "w") as f:
        f.write(f"""SELECTED OPERATING POINT (corrected; matches deployed web app and phase2.md)

Model: FT-UnivFD = frozen CLIP ViT-L/14 (open_clip, openai) + adversarial-aware
       LogisticRegression probe on RAW CLIP features (mlp_probe_adv_aware_raw.pkl, C=0.01).
Threshold: 0.30 (synthetic_prob > 0.30 => synthetic).
Two-model gate: DROPPED. FT-UnivFD is used alone. Wang2020 has ~0 FF++ recall and
       flags cloaked images as fake, so a gate would zero out detection.

Headline numbers (recomputed {Path(__file__).name}, validated vs phase2.md):
  FF++ AUC-ROC:                 {ff_auc:.4f}
  ProGAN AUC-ROC:               {progan_auc:.4f}
  FF++ recall @ 0.30:           {recall_at(0.3):.4f}
  Cloaked-image FPR @ 0.30:     {fpr03:.4f}  (400 eval cloaks: 200 ArcFace + 200 FaceNet)
  Cloaked-image FPR @ 0.40:     {fpr04:.4f}
  Cloaked-image FPR @ 0.30, ArcFace-only: {arc03:.4f}

Conflict status: PARTIALLY MITIGATED, not eliminated. Adversarial-aware retraining
  cuts cloaked FPR from ~0.95 (naive probe) to {fpr03:.3f} at t=0.30. The <0.15 target
  is met only at t>=0.40 (FPR {fpr04:.3f}, recall {recall_at(0.4):.3f}). t=0.30 is chosen to
  preserve recall ({recall_at(0.3):.3f}); the residual FPR is reported honestly.

Web-app mitigation: /protect applies light JPEG (Q=80) to the cloaked crop, which
  smooths the PGD perturbation. This keeps ArcFace ASR ~0.99 while cutting the
  ArcFace-cloak forensic FPR to ~0.02 (phase2.md Section 9).

Selection rationale:
  - FT-UnivFD adv-aware alone is the deployed configuration (no gate).
  - Other models fail the conflict at all thresholds (Wang2020 FPR 0.74,
    UnivFD k-NN min FPR 0.39); see conflict_summary.csv.
  - The earlier "0% cloaked FPR / fully resolved with a two-model gate" result was a
    normalization artifact (phase2.md Section 8) and has been retracted.
""")
    print(f"wrote {op.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
