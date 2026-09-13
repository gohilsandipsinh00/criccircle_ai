from fastapi import APIRouter
import onnxruntime as ort
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    providers = ort.get_available_providers()
    gpu_available = "CUDAExecutionProvider" in providers
    return {
        "status": "healthy",
        "service": "CricCircle AI Service",
        "version": settings.app_version,
        "gpu_available": gpu_available,
        "gpu_name": "CUDA (onnxruntime)" if gpu_available else "CPU mode",
        "onnx_providers": providers,
    }


@router.get("/")
async def root():
    return {
        "service": "CricCircle AI Highlight Service",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
