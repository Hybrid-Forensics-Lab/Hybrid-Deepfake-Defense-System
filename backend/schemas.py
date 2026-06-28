"""Pydantic request/response models. Annotations here are required by Pydantic."""

from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    gpu: bool
    ff_auc: float
    progan_auc: float


class DetectResponse(BaseModel):
    label: str            # "authentic" | "synthetic"
    confidence: float     # confidence in the returned label, [0.0, 1.0]
    model: str            # "ft_univfd" | "wang2020"
    conflict_warning: bool
    processing_time_ms: int


class ProtectResponse(BaseModel):
    cloaked_image_b64: str     # "data:image/png;base64,..."
    forensic_label: str        # forensic verdict on the cloaked image
    forensic_confidence: float
    conflict_warning: bool
    ssim: float
    psnr: float
    processing_time_ms: int
