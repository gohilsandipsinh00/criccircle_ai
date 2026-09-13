from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config import settings
from app.utils.logger import log
from app.api.routes import video, health, highlights, webhook, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        f"Starting CricCircle AI Service v"
        f"{settings.app_version}"
    )
    log.info(
        f"Environment: {settings.app_env}"
    )
    yield
    log.info("Shutting down CricCircle AI Service")


app = FastAPI(
    title="CricCircle AI Service",
    description=(
        "AI-powered cricket highlight detection service. "
        "Processes match videos to detect SIX, FOUR, "
        "WICKET, CATCH, and CELEBRATION moments."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(video.router)
app.include_router(highlights.router)
app.include_router(webhook.router)
app.include_router(upload.router)

# Serves local-disk raw uploads + "S3" outputs when use_s3=False
if not settings.use_s3:
    Path(settings.local_data_dir).mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media",
        StaticFiles(directory=settings.local_data_dir),
        name="media",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )
