"""Forensic inference engine: FT-UnivFD (CLIP ViT-L/14 + adv-aware probe).

Wraps the CLIP-feature + probe inference for single-image use. The probe is the
corrected adversarial-aware logistic regression trained on raw CLIP features, which
both detects deepfakes and resolves the cloaking conflict on its own (no Wang2020
gate — see phase2.md Section 8). Models load once at startup and stay on the GPU.
"""

import joblib
import open_clip
import torch

import config


class ForensicEngine:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # --- CLIP ViT-L/14 backbone (frozen) ---
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device).eval()

        # --- adversarial-aware logistic-regression probe (raw-feature) ---
        self.probe = joblib.load(config.PROBE_PATH)

    def predict_ft_univfd(self, pil_image):
        """Return P(synthetic) in [0, 1] from CLIP features + probe.

        The probe was trained on RAW (un-normalized) CLIP features (L2 norm ~20).
        Do NOT L2-normalize here, or every score collapses to ~0.014.
        """
        img = pil_image.convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                feats = self.clip_model.encode_image(x)
        prob = float(self.probe.predict_proba(feats.float().cpu().numpy())[:, 1][0])
        return prob

    def classify(self, pil_image, threshold=None):
        """Return (label, confidence-in-label, synthetic_prob)."""
        if threshold is None:
            threshold = config.DETECT_THRESHOLD
        score = self.predict_ft_univfd(pil_image)
        if score > threshold:
            return "synthetic", score, score
        return "authentic", 1.0 - score, score
