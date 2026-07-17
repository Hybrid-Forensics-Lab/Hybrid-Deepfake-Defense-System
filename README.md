# Hybrid Deepfake Defense System

A dual-layer system for **media authentication** and **identity protection**. It detects AI-generated images and cloaks real faces against unauthorized recognition, while keeping the two defenses from undermining each other.

**Live demo:** http://146.148.65.227

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.137-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)

---

## Overview

Digital media defenses usually live in two separate worlds:

- **Forensic detectors** analyze an image to decide whether it was synthetically generated.
- **Adversarial cloaking tools** add small perturbations to real photographs so that face recognition systems fail to extract a usable identity.

These two goals collide. The pixel-level perturbations added by cloaking resemble the artifacts that forensic detectors are trained to flag, so a legitimately protected real photo can be misclassified as synthetic. This project studies that conflict directly and ships a single system that performs both tasks and measures how much they interfere.

The system has two layers:

1. **Forensics layer.** A fine-tuned UnivFD detector (frozen CLIP ViT-L/14 backbone with an adversarial-aware probe head) classifies an image as authentic or synthetic.
2. **Privacy layer.** An untargeted PGD attack cloaks a detected face against ArcFace and, optionally, FaceNet, so recognition fails while the image stays visually close to the original.

The forensic probe is trained to be **adversarial-aware**: cloaked real images are added to its training set as authentic examples, which substantially reduces the false-positive rate on cloaked photos.

---

## Key Results

All numbers below are the final, verified figures from the Phase 2 results log (`phase2.md`).

### Forensic detection (FT-UnivFD, deployed configuration)

| Metric | Value |
|---|---|
| FF++ AUC-ROC | 0.812 |
| ProGAN AUC-ROC | 0.829 |
| FF++ recall at operating threshold (0.30) | 0.620 |
| False-positive rate on cloaked real images (t=0.30) | 0.177 (0.122 at t=0.40) |

Deployed detector: frozen CLIP ViT-L/14 plus a logistic-regression probe (C=0.01) on **raw** 768-dimensional CLIP features, operating threshold 0.30, no secondary gate.

### Identity cloaking (untargeted PGD, epsilon=0.03, 40 steps)

| Recognizer | ASR | SSIM | PSNR |
|---|---|---|---|
| ArcFace | 1.000 | 0.902 | 34.21 dB |
| FaceNet | 1.000 | 0.863 | 33.54 dB |

ASR is the attack success rate (fraction of faces whose identity match is broken). SSIM and PSNR measure how visually close the cloaked image stays to the original.

### Conflict mitigation

| Setting | Cloaked-image FPR |
|---|---|
| Naive probe (no adversarial-aware training) | about 0.95 |
| Adversarial-aware probe (raw features), t=0.30 | 0.177 |
| Adversarial-aware probe plus JPEG Q=80 on the cloaked output (ArcFace cloaks) | 0.020 |

Applying a light JPEG re-compression (quality 80) to the cloaked face smooths the high-frequency PGD pattern. This keeps ArcFace ASR at about 0.99 while cutting the forensic false-positive rate from 6 percent to 2 percent on the ArcFace cloak set. This step is wired into the web app `/protect` endpoint.

The conflict is **mitigated, not eliminated**. Untargeted-PGD artifacts remain partially confusable with synthetic artifacts in CLIP feature space, which is documented as an open limitation.

---

## System Architecture

```
                         Input image (single frame)
                                    |
                          MTCNN face detection
                                    |
                ____________________|____________________
               |                                         |
        Forensics layer                            Privacy layer
   (detect synthetic media)                  (cloak a real identity)
               |                                         |
   Frozen CLIP ViT-L/14                      Untargeted PGD (L-inf, eps=0.03)
   + adversarial-aware probe                 against ArcFace (and optional FaceNet)
   (raw features, threshold 0.30)            + JPEG Q=80 on the cloaked crop
               |                                         |
        authentic / synthetic                   cloaked image returned,
        + confidence                            re-checked by the forensic probe
               |_____________________  _________________|
                                     ||
                              FastAPI backend (:8000)
                              React frontend (nginx :80)
                              Live at http://146.148.65.227
```

---

## Web Application

The demo deliverable exposes both layers through a REST API and a browser UI.

- **Backend.** FastAPI plus uvicorn, single worker with lifespan model loading, served as a systemd service (`deepfake-api`) on port 8000. Models load once at startup (CLIP plus probe, ArcFace, MTCNN); FaceNet loads lazily only when requested.
- **Frontend.** Vite plus React 19, built to static assets and served by nginx on port 80. The interface offers a Detect mode and a Protect mode, with drag-and-drop upload, live result animations, and the headline metrics shown from `/health`.

### API endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/health` | GET | none | status, models loaded, GPU flag, FF++ and ProGAN AUC |
| `/detect` | POST | multipart `image` (PNG or JPEG, max 10 MB) | label, confidence, model, processing time |
| `/protect` | POST | multipart `image` plus optional `epsilon`, `use_facenet`, `jpeg_quality` | base64 cloaked PNG, forensic label, SSIM, PSNR, processing time |

Interactive API docs are available at `/docs` (Swagger UI). Screenshots for the report live in `docs/figures/swagger/` and `docs/figures/webapp/`.

---

## Repository Structure

```
.
├── ml_forensics/             # Forensic detection
│   ├── evaluate/             # Inference wrappers + conflict matrix
│   │   ├── wang2020_inference.py
│   │   ├── gragnaniello2021_inference.py
│   │   ├── xception_inference.py
│   │   ├── univfd_inference.py
│   │   └── run_conflict_matrix.py
│   ├── finetune/             # Probe training (linear, MLP, adversarial-aware)
│   │   ├── extract_clip_features.py
│   │   ├── train_linear_probe.py
│   │   ├── train_mlp_probe.py
│   │   └── train_adv_aware_probe.py
│   └── models/               # Cloned model repos and weights (gitignored)
├── ml_privacy/               # Adversarial cloaking
│   ├── attacks/              # PGD and Fawkes implementations
│   ├── evaluate/             # Untargeted PGD, ASR / SSIM / PSNR, robustness
│   │   ├── pgd_untargeted_arcface.py
│   │   ├── pgd_untargeted_facenet.py
│   │   ├── eval_robustness_asr.py
│   │   └── apply_robustness_transforms.py
│   └── models/               # ArcFace iresnet weights (gitignored)
├── backend/                  # FastAPI inference server
│   ├── main.py               # Endpoints and app wiring
│   ├── config.py             # Paths, thresholds, hyperparameters
│   ├── schemas.py            # Pydantic request/response models
│   └── models/               # ForensicEngine and CloakEngine wrappers
├── frontend/                 # Vite + React single-page app
│   ├── src/                  # App, components, api client, animations
│   └── public/               # Favicon and static assets
├── deploy/                   # systemd unit, nginx config, deploy.sh
├── results/                  # Result CSVs (large binaries gitignored)
├── data/                     # Dataset manifests tracked, images gitignored
├── docs/                     # Sprint plan, report figures
├── report/                   # Midterm report PDF, report figures
└── environments/             # Conda environment files
```

---

## Technology Stack

- **Deep learning:** PyTorch 2.1.0 (CUDA 12.1 build), torchvision, open-clip-torch
- **Face processing:** facenet-pytorch (MTCNN, FaceNet, InceptionResnetV1), ArcFace iresnet100
- **Evaluation:** scikit-learn, scikit-image, NumPy, pandas
- **Backend:** FastAPI, uvicorn, Pydantic v2, Python 3.10
- **Frontend:** React 19, Vite, axios
- **Serving:** systemd plus nginx on an Ubuntu 22.04 host
- **Hardware:** GCP VM (g2-standard-8, 8 vCPU, 32 GB RAM) with an NVIDIA L4 24 GB GPU

---

## Setup

### Prerequisites

- Ubuntu 22.04 with NVIDIA drivers and the CUDA 12.1 toolkit
- Miniconda or Anaconda
- Node.js 20 or newer (for the frontend)
- An NVIDIA GPU with at least 12 GB of VRAM is recommended

### Conda environments

The project uses separate conda environments so that dependencies never clash. Always run module scripts with `conda run -n <env> python ...` rather than relying on shell activation.

| Environment | Used for | Dependency spec |
|---|---|---|
| `forensics-env` | everything in `ml_forensics/` | documented in `phase2.md` |
| `privacy-env` | everything in `ml_privacy/` | `environments/privacy-env.yml` |
| `webapp-env` | the FastAPI backend | `backend/requirements.txt` |
| `fawkes-env` | the isolated TensorFlow-based Fawkes baseline | standalone |

```bash
git clone git@github.com:Hybrid-Forensics-Lab/Hybrid-Deepfake-Defense-System.git
cd Hybrid-Deepfake-Defense-System

# Privacy environment
conda env create -f environments/privacy-env.yml

# Verify GPU access
conda run -n privacy-env python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Datasets

Datasets live under `data/` and are gitignored. Only the CSV manifests are tracked.

| Dataset | Path | Manifest |
|---|---|---|
| ProGAN (synthetic vs real) | `data/progan/` | `data/progan/manifest.csv` |
| FaceForensics++ (face manipulations) | `data/ff_plus_plus/` | per-resolution manifests under `test/` |
| LFW (real faces for cloaking) | `data/lfw/` | `data/lfw/manifest.csv` |

Download instructions are in the sprint plan under `docs/`.

---

## Running

### Forensics

All forensic scripts use argparse. Run any of them with `--help` for the exact flags.

```bash
# Baseline detector inference on the ProGAN test set
conda run -n forensics-env python ml_forensics/evaluate/univfd_inference.py --help

# Train the deployed adversarial-aware probe (raw CLIP features)
conda run -n forensics-env python ml_forensics/finetune/train_adv_aware_probe.py --help
```

### Privacy

```bash
# Untargeted PGD cloaking against ArcFace
conda run -n privacy-env python ml_privacy/evaluate/pgd_untargeted_arcface.py --help

# Robustness evaluation under JPEG and downsampling transforms
conda run -n privacy-env python ml_privacy/evaluate/eval_robustness_asr.py --help
```

### Backend

```bash
cd backend
conda run -n webapp-env uvicorn main:app --host 0.0.0.0 --port 8000
# Swagger UI at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # local dev server
npm run build    # production build into dist/
```

### Deploy

A single script rebuilds the frontend, publishes it to nginx, and restarts the API service.

```bash
bash deploy/deploy.sh
```

---

## Results and Reproducibility

- **`phase2.md`** is the authoritative Phase 2 results log. It records every experiment with metrics, dataset details, environment notes, and timing, including the normalization-artifact correction that produced the honest final numbers.
- **`results/`** holds the per-image and aggregate CSVs. Plots and figures are generated from these CSVs.
- **`docs/figures/`** holds report figures, plus Swagger and UI screenshots of the live app.

---

## Limitations

- The cloaking and detection conflict is mitigated but not fully resolved. At the operating threshold of 0.30, about 18 percent of cloaked real images are still flagged as synthetic. Raising the threshold lowers this rate at the cost of recall.
- ArcFace cloaking is fragile beyond light compression. JPEG quality 50 and bilinear resampling collapse ArcFace ASR, whereas FaceNet cloaks remain robust to the same transforms.
- The system processes single still images only. No video, audio, or real-time processing.

---

## Team

| Member | Responsibility |
|---|---|
| **Mohammad Raafe** | Forensics model development, inference pipeline, testing, and quantitative evaluation |
| **Wagd Haroon** | Privacy dataset collection and preprocessing; report writing and compilation |
| **Saeed Ahmed** | Forensic dataset collection and preprocessing; FastAPI backend; React frontend |
| **Fatima Bey** | Privacy model development, cloaking pipeline, testing, and quantitative evaluation |

---

## Timeline

| Deliverable | Date |
|---|---|
| Phase 2 ML results (internal) | June 23, 2026 |
| Web app demo-ready (internal) | June 26, 2026 |
| Final report (target) | June 28, 2026 |
| Poster and hard deadline | June 30, 2026 |
| Final presentation | July 1, 2026 |

**Supervisor:** Dr. Sahil Garg

---

## License

Developed as part of BCS 410-1 (Computer Science Project) at Canadian University Dubai. Code and documentation are for academic use.
