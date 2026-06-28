"""Cloaking engine: MTCNN face detection + untargeted PGD against ArcFace
(optionally FaceNet). Wraps the attack logic from
ml_privacy/evaluate/pgd_untargeted_arcface.py for single-image use.

The detected face is cloaked at the recognizer's native resolution, then composited
back into the original image so the demo shows a natural before/after.
"""

import io
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import config

sys.path.insert(0, str(config.REPO_ROOT))
from ml_privacy.models.arcface.iresnet import iresnet100

ARCFACE_SIZE = 112
FACENET_SIZE = 160
NORMALIZE = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0,1] -> [-1,1]


class NoFaceError(Exception):
    pass


def compute_ssim_psnr(orig_np, cloaked_np):
    orig_f = orig_np.astype(np.float64) / 255.0
    cloaked_f = cloaked_np.astype(np.float64) / 255.0
    ssim_val = structural_similarity(orig_f, cloaked_f, channel_axis=2, data_range=1.0)
    psnr_val = peak_signal_noise_ratio(orig_f, cloaked_f, data_range=1.0)
    return float(ssim_val), float(psnr_val)


class CloakEngine:
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        # --- ArcFace iresnet100 ---
        self.arcface = iresnet100(pretrained=False, fp16=False)
        ckpt = torch.load(config.ARCFACE_WEIGHTS, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.arcface.load_state_dict(state, strict=False)
        self.arcface = self.arcface.to(self.device).eval()

        self.facenet = None  # loaded lazily only if FaceNet cloaking requested

    # --- embeddings ---
    def _embed_arcface(self, x):
        return F.normalize(self.arcface(NORMALIZE(x)), p=2, dim=-1)

    def _embed_facenet(self, x):
        return self.facenet(NORMALIZE(x))  # InceptionResnetV1 output is already L2-normed

    def _ensure_facenet(self):
        if self.facenet is None:
            from facenet_pytorch import InceptionResnetV1
            self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    # --- PGD (untargeted, L-inf, fp32) ---
    def _pgd_arcface(self, x_orig, epsilon, steps, alpha):
        """Minimise cosine similarity to the clean ArcFace embedding."""
        x_orig = x_orig.detach()
        with torch.no_grad():
            e_orig = self._embed_arcface(x_orig)
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = -F.cosine_similarity(self._embed_arcface(x_adv), e_orig.detach(), dim=-1).sum()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
                x_adv = (x_orig + delta).clamp(0, 1)
        return x_adv.detach()

    def _pgd_facenet(self, x_orig, epsilon, steps, alpha):
        """Maximise L2 distance from the clean FaceNet embedding."""
        x_orig = x_orig.detach()
        with torch.no_grad():
            e_orig = self._embed_facenet(x_orig)
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = torch.linalg.vector_norm(
                self._embed_facenet(x_adv) - e_orig.detach(), ord=2, dim=-1).sum()
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
                x_adv = (x_orig + delta).clamp(0, 1)
        return x_adv.detach()

    # --- face detection ---
    def _detect_crop(self, pil_image):
        img = pil_image.convert("RGB")
        boxes, _ = self.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            raise NoFaceError("No face detected")
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[0]]
        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise NoFaceError("Invalid face bounding box")
        return img, img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

    def _jpeg_recompress(self, pil_img, quality):
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def cloak(self, pil_image, epsilon, steps, use_facenet=False, jpeg_quality=None):
        """Detect a face, cloak it, composite back. Returns (PIL image, ssim, psnr)."""
        orig_img, crop, box = self._detect_crop(pil_image)
        alpha = epsilon / steps * config.PGD_ALPHA_FACTOR

        # ArcFace cloaking at 112x112
        to_arc = T.Compose([T.Resize((ARCFACE_SIZE, ARCFACE_SIZE)), T.ToTensor()])
        x = to_arc(crop).unsqueeze(0).to(self.device)
        x_adv = self._pgd_arcface(x, epsilon, steps, alpha)
        cloaked_np = (x_adv[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        cloaked_crop = Image.fromarray(cloaked_np)

        # Optional sequential FaceNet cloaking at 160x160
        if use_facenet:
            self._ensure_facenet()
            to_fn = T.Compose([T.Resize((FACENET_SIZE, FACENET_SIZE)), T.ToTensor()])
            xf = to_fn(cloaked_crop).unsqueeze(0).to(self.device)
            xf_adv = self._pgd_facenet(xf, epsilon, steps, alpha)
            cloaked_np = (xf_adv[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            cloaked_crop = Image.fromarray(cloaked_np)

        # Light JPEG re-compression smooths the PGD perturbation: keeps attack success
        # high (~0.99 ASR) while making the cloak less detectable by the forensic probe
        # (cloaked FPR 6% -> 2% at Q=80; phase2.md Section 9).
        if jpeg_quality:
            cloaked_crop = self._jpeg_recompress(cloaked_crop, jpeg_quality)
            cloaked_np = np.array(cloaked_crop)

        # SSIM/PSNR of the final cloaked crop vs the clean face crop (same size)
        clean_ref = np.array(crop.resize(cloaked_crop.size, Image.BICUBIC).convert("RGB"))
        ssim_val, psnr_val = compute_ssim_psnr(clean_ref, cloaked_np)

        # Composite the cloaked face back into the original image
        result = orig_img.copy()
        result.paste(cloaked_crop.resize((box[2] - box[0], box[3] - box[1]), Image.BICUBIC),
                     (box[0], box[1]))
        return result, ssim_val, psnr_val
