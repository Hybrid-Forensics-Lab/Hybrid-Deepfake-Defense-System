# Hybrid Deepfake Defense System 🛡️

A comprehensive framework integrating **Generative Image Forensics** and **Adversarial Privacy Protection**. This project provides a dual-layer defense system designed to identify manipulated media and proactively protect user imagery from unauthorized AI exploitation.

---

## 🚀 Overview

In an era of hyper-realistic generative AI, this system addresses the trust gap in digital media through two distinct mechanisms:

1. **Passive Defense (Forensics):** Utilizing frequency-domain analysis (Fast Fourier Transform) and CNN-based classifiers to detect traces left by GAN and Diffusion-based generators.
2. **Active Defense (Privacy):** Implementing adversarial filters (PGD-based) that add imperceptible noise to images, "breaking" the ability of deepfake models and facial recognition systems to process them accurately.

## 🛠️ Technology Stack

- **Deep Learning:** PyTorch, Torchvision
- **Backend:** FastAPI (Python 3.10+)
- **Frontend:** React.js (TypeScript)
- **Environment:** Ubuntu 22.04 LTS (via WSL 2)
- **Hardware Target:** NVIDIA RTX 40-series (CUDA 12.x optimized)

## 📂 Project Structure

- `ml_forensics/` - Detection models and frequency analysis scripts.
- `ml_privacy/` - Adversarial noise generation and privacy filter implementation.
- `backend/` - FastAPI service for processing image uploads.
- `frontend/` - React dashboard for real-time analysis and visualization.
- `data/` - (_Git ignored_) Local datasets for benchmarking and training.
- `docs/` - Project documentation, including the Synopsis and Midterm reports.

## 👥 The Team

- **Mohammad** - Project Lead / ML Infrastructure
- **Wagd** - [Role TBD]
- **Saeed** - [Role TBD]
- **Fatima** - [Role TBD]

## 📝 Setup for Team Members

To ensure a consistent development environment, please follow these steps:

1. **Environment:** Ensure you are working within **WSL 2 (Ubuntu)**.
2. **Clone the Repo:**
   ```bash
   git clone git@github.com:Hybrid-Forensics-Lab/Hybrid-Deepfake-Defense-System.git
   ```
3. **Checkout Development Branch:**
   ```bash
   git checkout develop
   ```
4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
