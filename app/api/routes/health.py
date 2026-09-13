from fastapi import APIRouter
import torch
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "CricCircle AI Service",
        "version": settings.app_version,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU mode"
        ),
    }


@router.get("/")
async def root():
    return {
        "service": "CricCircle AI Highlight Service",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
