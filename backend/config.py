"""Central configuration for the Hybrid Deepfake Defense System backend.

All paths, thresholds and hyperparameters live here. No hardcoding in main.py
or the model wrappers. The operating point below is the locked Phase 2 result.
"""

from pathlib import Path

# Repo root is one level up from backend/
REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Forensics ---
# Corrected adv-aware probe trained on consistent RAW CLIP features (the original
# mlp_probe_adv_aware.pkl had a normalization artifact; see phase2.md Section 8).
PROBE_PATH = REPO_ROOT / "results/phase2/forensics/mlp_probe_adv_aware_raw.pkl"

# --- Privacy / cloaking ---
ARCFACE_WEIGHTS = REPO_ROOT / "ml_privacy/models/arcface/ms1mv3_arcface_r100_fp16.pth"

# --- Operating point (corrected Phase 2) ---
# FT-UnivFD adv-aware probe alone; no Wang2020 gate (Wang2020 has ~0 FF++ recall and
# flags cloaked images as fake, so it would zero out detection — see phase2.md Section 8).
DETECT_THRESHOLD = 0.30        # FT-UnivFD synthetic-probability threshold

PGD_EPSILON = 0.03             # untargeted PGD L-inf budget (default)
PGD_STEPS = 40
PGD_ALPHA_FACTOR = 2.0         # alpha = epsilon / steps * alpha_factor
CLOAK_FACENET_DEFAULT = False  # default /protect cloaks ArcFace only
PROTECT_JPEG_QUALITY = 80      # light JPEG on cloaked output: ASR ~0.99, cloaked FPR 6%->2% (phase2.md S9)

# --- Upload limits ---
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

# --- Reported performance numbers (for /health and report) ---
FF_AUC = 0.812
PROGAN_AUC = 0.829

# --- Runtime ---
DEVICE = "cuda"
CORS_ORIGINS = ["*"]
