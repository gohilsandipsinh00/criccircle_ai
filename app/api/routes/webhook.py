from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from app.api.dependencies import verify_webhook_secret
from app.utils.logger import log

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


class NestJSUploadCompleteEvent(BaseModel):
    """Fired by the NestJS backend once a raw video finishes
    uploading to S3, to kick off AI processing."""
    video_id: str
    user_id: str
    match_id: Optional[str] = None
    raw_video_s3_key: str
    raw_video_url: str
    user_fcm_token: Optional[str] = None


@router.post(
    "/nestjs/upload-complete",
    dependencies=[Depends(verify_webhook_secret)],
)
async def handle_upload_complete(event: NestJSUploadCompleteEvent):
    """
    Handler for NestJS -> AI service webhooks. Kept separate from
    /api/v1/videos/process so NestJS has one stable, secret-guarded
    entry point independent of the direct processing API.
    """
    log.info(
        f"Received upload-complete webhook for video {event.video_id}"
    )
    return {
        "success": True,
        "message": "Webhook received",
        "video_id": event.video_id,
    }
