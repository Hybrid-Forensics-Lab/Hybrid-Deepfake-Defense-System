"""FastAPI app for the Hybrid Deepfake Defense System.

Endpoints:
  GET  /health   -> model + GPU status and headline metrics
  POST /detect   -> forensic classification via the two-model conflict gate
  POST /protect  -> identity cloaking (untargeted PGD) + forensic verdict

Models are loaded once at startup (single uvicorn worker). Sync endpoints run in
FastAPI's threadpool, so GPU work does not block the event loop.
"""

import base64
import io
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import config
import schemas
from models.forensics import ForensicEngine
from models.privacy import CloakEngine, NoFaceError

engines = {}


@asynccontextmanager
async def lifespan(app):
    print("Loading models (CLIP + probe + Wang2020 + ArcFace + MTCNN)...")
    engines["forensic"] = ForensicEngine(config.DEVICE)
    engines["cloak"] = CloakEngine(config.DEVICE)
    print("Models loaded. API ready.")
    yield
    engines.clear()
    torch.cuda.empty_cache()


app = FastAPI(title="Hybrid Deepfake Defense System API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(upload):
    if upload.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {upload.content_type}")
    data = upload.file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400,
                            detail=f"File exceeds {config.MAX_UPLOAD_MB}MB limit")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")


@app.get("/health", response_model=schemas.HealthResponse)
def health():
    return schemas.HealthResponse(
        status="ok",
        models_loaded=["ft_univfd", "arcface"],
        gpu=torch.cuda.is_available(),
        ff_auc=config.FF_AUC,
        progan_auc=config.PROGAN_AUC,
    )


@app.post("/detect", response_model=schemas.DetectResponse)
def detect(image: UploadFile = File(...)):
    img = _read_image(image)
    t0 = time.time()
    label, conf, _ = engines["forensic"].classify(img)
    dt = int((time.time() - t0) * 1000)
    return schemas.DetectResponse(
        label=label,
        confidence=round(conf, 4),
        model="ft_univfd",
        conflict_warning=False,
        processing_time_ms=dt,
    )


@app.post("/protect", response_model=schemas.ProtectResponse)
def protect(image: UploadFile = File(...),
            epsilon: float = Form(None),
            use_facenet: bool = Form(None),
            jpeg_quality: int = Form(None)):
    img = _read_image(image)
    eps = epsilon if epsilon is not None else config.PGD_EPSILON
    use_fn = use_facenet if use_facenet is not None else config.CLOAK_FACENET_DEFAULT
    jq = jpeg_quality if jpeg_quality is not None else config.PROTECT_JPEG_QUALITY

    t0 = time.time()
    try:
        cloaked_img, ssim_val, psnr_val = engines["cloak"].cloak(
            img, eps, config.PGD_STEPS, use_fn, jq)
    except NoFaceError:
        raise HTTPException(status_code=400,
                            detail="No face detected in the uploaded image")

    label, conf, _ = engines["forensic"].classify(cloaked_img)
    # conflict_warning = the cloak failed to read as authentic (residual cloaked-FPR case)
    warn = label == "synthetic"

    buf = io.BytesIO()
    cloaked_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    dt = int((time.time() - t0) * 1000)

    return schemas.ProtectResponse(
        cloaked_image_b64="data:image/png;base64," + b64,
        forensic_label=label,
        forensic_confidence=round(conf, 4),
        conflict_warning=warn,
        ssim=round(ssim_val, 4),
        psnr=round(psnr_val, 2),
        processing_time_ms=dt,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
