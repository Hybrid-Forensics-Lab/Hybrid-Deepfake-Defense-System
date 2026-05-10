# Hybrid Deepfake Defense System

A dual-layer framework for **media authentication** and **identity protection**, designed to detect AI-generated images and protect real photographs from unauthorized facial recognition — without the two defenses undermining each other.

## The Problem

Current digital media defenses exist in two isolated tracks. Forensic detectors analyze images to determine if they are synthetically generated. Adversarial cloaking tools modify real images to prevent facial recognition systems from extracting biometric identity. These two goals conflict: the pixel-level perturbations added by cloaking tools resemble the artifacts that forensic detectors are trained to flag, causing legitimately protected real photographs to be misclassified as synthetic.

This project targets that conflict directly.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Input Image                        │
│              (static, single frame)                  │
└──────────────────────┬──────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │   MTCNN Alignment   │
            │  & Preprocessing    │
            └──────────┬──────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
  ┌───────▼───────┐       ┌────────▼────────┐
  │   Forensics   │       │    Privacy      │
  │    Module     │       │    Module       │
  │               │       │                 │
  │  5 detectors  │       │  3+ cloaking    │
  │  (inference)  │       │  methods        │
  └───────┬───────┘       └────────┬────────┘
          │                         │
          └────────────┬────────────┘
                       │
            ┌──────────▼──────────┐
            │  Conflict Matrix    │
            │  (FPR < 15%)        │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  FastAPI Backend    │
            │  + React Frontend   │
            └─────────────────────┘
```

## Project Structure

```
├── ml_forensics/          # Forensic detection models and evaluation
│   ├── models/            # Cloned model repos and weight files
│   ├── data/              # Preprocessing and data loading scripts
│   └── evaluate/          # Inference wrappers and metric computation
├── ml_privacy/            # Adversarial cloaking pipeline
│   ├── attacks/           # PGD, Fawkes, and other cloaking implementations
│   ├── data/              # LFW preprocessing and data loading
│   └── evaluate/          # ASR, SSIM, PSNR metric computation
├── backend/               # FastAPI inference server (Phase 2)
├── frontend/              # React.js dashboard (Phase 2)
├── results/               # Benchmark results, CSVs, and figures
│   ├── forensics/
│   ├── privacy/
│   └── conflict/
├── environments/          # Conda environment YAML files
├── docs/                  # Reports, presentations, and synopsis
├── data/                  # (gitignored) Local datasets
├── .gitignore
├── requirements.txt
└── README.md
```

## Scope

- **Images only.** No video, no audio, no real-time processing.
- **Inference only (Phase 1).** Pre-trained models evaluated as-is; no retraining.
- **Local execution.** All inference runs on consumer GPU hardware. No cloud compute, no paid APIs.

## Forensic Models Under Evaluation

| Model | Paper | Architecture | Target Artifacts |
|---|---|---|---|
| Wang2020 (CNNDetection) | Wang et al., CVPR 2020 | ResNet-50 | Cross-generator GAN artifacts |
| Gragnaniello2021 | Gragnaniello et al., ICME 2021 | ResNet-50 (no downsampling) | Frequency-domain GAN artifacts |
| Mandelli2022 | Mandelli et al., ICIP 2022 | Multi-CNN ensemble | Noise residuals and camera fingerprints |
| UnivFD | Ojha et al., CVPR 2023 | CLIP ViT-L/14 + linear probe | Cross-generator (incl. diffusion) |
| Xception (FF++) | Chollet 2017 / Rössler et al. 2019 | Xception | FF++ face manipulations |

## Cloaking Methods Under Evaluation

| Method | Paper | Approach |
|---|---|---|
| PGD | Madry et al., ICLR 2018 | Untargeted L∞ perturbation against face embeddings |
| Fawkes | Shan et al., USENIX Security 2020 | Targeted embedding-space cloaking |
| LowKey | Cherepanova et al., ICLR 2021 | Black-box adversarial attack on recognition pipelines |

**Victim recognizers:** ArcFace (primary), FaceNet (secondary).

## Performance Targets

| Objective | Metric | Target |
|---|---|---|
| Forensic detection accuracy | Accuracy | ≥ 90% |
| Forensic detection ranking | AUC-ROC | > 0.90 |
| Privacy: recognition evasion | ASR | ≥ 80% |
| Privacy: image quality | SSIM | ≥ 0.90 |
| Privacy: image quality | PSNR | ≥ 30 dB |
| Conflict resolution | Forensic FPR on cloaked images | < 15% |

## Technology Stack

- **Deep Learning:** PyTorch 2.1, torchvision, open-clip-torch
- **Face Processing:** facenet-pytorch (MTCNN, FaceNet), insightface (ArcFace)
- **Evaluation:** scikit-learn, scikit-image, matplotlib, pandas
- **Backend (Phase 2):** FastAPI, Python 3.10+
- **Frontend (Phase 2):** React.js, TypeScript
- **Hardware:** NVIDIA RTX 4060 (8 GB VRAM), CUDA 12.x
- **Environment:** Ubuntu 22.04 LTS (WSL 2)

## Setup

### Prerequisites
- WSL 2 with Ubuntu 22.04
- NVIDIA drivers + CUDA 12.x toolkit
- Miniconda or Anaconda

### Installation

```bash
# Clone the repo
git clone git@github.com:Hybrid-Forensics-Lab/Hybrid-Deepfake-Defense-System.git
cd Hybrid-Deepfake-Defense-System
git checkout develop

# Create the appropriate conda environment
conda env create -f environments/forensics-env.yml   # For forensics work
conda env create -f environments/privacy-env.yml     # For privacy/cloaking work

# Activate and verify GPU access
conda activate forensics-env
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Datasets

Datasets are stored locally in `data/` (gitignored). See the sprint plan in `docs/` for download instructions and links.

## Team

| Member | Responsibility |
|---|---|
| **Mohammad Raafe** | Forensics model development, inference pipeline, testing, and quantitative evaluation |
| **Wagd Haroon** | Privacy dataset collection and preprocessing; report writing and compilation |
| **Saeed Ahmed** | Forensic dataset collection and preprocessing; FastAPI backend; React.js frontend |
| **Fatima Bey** | Privacy model development, cloaking pipeline, testing, and quantitative evaluation |

## Project Timeline

| Phase | Deliverable | Deadline |
|---|---|---|
| Phase 1 | Midterm Report — benchmarking and model selection | May 21, 2026 |
| Phase 2 | Final Report — optimization, conflict matrix, web app | June 22, 2026 |
| Presentation | Final Presentation and poster | June 24, 2026 |

**Supervisor:** Dr. Sahil Garg

## License

This project is developed as part of BCS 410-1 (Computer Science Project) at Ajman University. Code and documentation are for academic use.
