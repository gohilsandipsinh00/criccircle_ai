import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.services.s3_service import S3Service
from app.utils.logger import log

router = APIRouter(prefix="/api/v1/videos", tags=["upload"])

s3_service = S3Service()


class UploadUrlRequest(BaseModel):
    user_id: str
    filename: str


@router.post("/upload-url")
async def get_upload_url(request: UploadUrlRequest):
    """
    Returns where the client should upload the raw video.

    - use_s3=True: a presigned S3 PUT URL, exactly like production.
    - use_s3=False (demo default, no AWS creds yet): a local endpoint
      (POST /api/v1/videos/upload) plus an s3_key the client echoes
      back with the file.
    """
    safe_name = Path(request.filename).name
    s3_key = f"raw/{request.user_id}/{uuid.uuid4().hex}_{safe_name}"

    if settings.use_s3:
        upload_url = s3_service.generate_presigned_url(s3_key)
        raw_video_url = (
            f"https://{settings.s3_bucket_raw_videos}.s3."
            f"{settings.aws_region}.amazonaws.com/{s3_key}"
        )
        return {
            "mode": "s3",
            "upload_url": upload_url,
            "s3_key": s3_key,
            "raw_video_url": raw_video_url,
        }

    raw_video_url = f"{settings.public_base_url}/media/{s3_key}"
    return {
        "mode": "local",
        "upload_url": f"{settings.public_base_url}/api/v1/videos/upload",
        "s3_key": s3_key,
        "raw_video_url": raw_video_url,
    }


@router.post("/upload")
async def upload_video_local(
    s3_key: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Local-disk fallback for raw video upload, used when use_s3=False.
    Mirrors what a presigned S3 PUT would do: the file ends up
    addressable at the same s3_key handed out by /upload-url.
    """
    if settings.use_s3:
        raise HTTPException(
            status_code=400,
            detail="Local upload is disabled; use_s3=True is set",
        )

    dest = Path(settings.local_data_dir) / s3_key
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            await out.write(chunk)

    log.info(f"[local] Received upload {dest} ({size} bytes)")

    return {
        "success": True,
        "s3_key": s3_key,
        "raw_video_url": f"{settings.public_base_url}/media/{s3_key}",
        "file_size_bytes": size,
    }
