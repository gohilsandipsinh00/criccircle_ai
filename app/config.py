from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "CricCircle AI Service"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    port: int = 8001

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # AWS S3
    use_s3: bool = False
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "ap-south-1"
    s3_bucket_raw_videos: str = "criccircle-raw-videos"
    s3_bucket_highlights: str = "criccircle-highlights"
    s3_bucket_thumbnails: str = "criccircle-thumbnails"

    # Local storage fallback (used when use_s3=False and/or Postgres
    # isn't provisioned yet -- see app/services/local_store.py)
    use_local_store: bool = True
    local_data_dir: str = "./data"
    # Base URL the phone/emulator uses to reach this service's static
    # media (raw uploads + local "S3" outputs). Override for LAN/ngrok.
    public_base_url: str = "http://localhost:8001"

    # Firebase
    firebase_credentials_path: str = "./firebase_credentials.json"

    # NestJS Backend
    nestjs_backend_url: str = "http://localhost:3000"
    nestjs_webhook_secret: str

    # ML Models (ONNX Runtime, not torch -- see app/ml/models/*.py)
    yolo_model_path: str = "./models/yolo/cricket_yolo_v1.onnx"
    classifier_model_path: str = \
        "./models/classifier/cricket_classifier_v1.onnx"
    use_gpu: bool = False
    confidence_threshold: float = 0.65

    # Processing
    max_video_duration_seconds: int = 1800
    highlight_clip_duration: int = 6
    frames_per_second: int = 2
    audio_spike_threshold: float = 2.5
    max_concurrent_jobs: int = 3

    # Branding
    watermark_path: str = "./assets/criccircle_watermark.png"
    branding_enabled: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
