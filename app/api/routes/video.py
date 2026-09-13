import asyncio

from fastapi import (
    APIRouter, BackgroundTasks,
    Depends, HTTPException
)
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.core.video_processor import CricketVideoProcessor
from app.services.s3_service import S3Service
from app.services.fcm_service import FCMService
from app.services.webhook_service import WebhookService
from app.services.local_store import store
from app.utils.logger import log

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


class ProcessVideoRequest(BaseModel):
    video_id: str
    user_id: str
    match_id: Optional[str] = None
    raw_video_s3_key: str
    raw_video_url: str
    user_fcm_token: Optional[str] = None


processor = CricketVideoProcessor()
s3_service = S3Service()
fcm_service = FCMService()
webhook_service = WebhookService()


@router.post("/process")
async def process_video(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Start AI processing of a cricket match video.
    Processing runs in background.
    User receives FCM notification when done.
    """
    log.info(
        f"Received process request for video "
        f"{request.video_id}"
    )

    store.save_status(
        video_id=request.video_id,
        status="queued",
        progress_percentage=0,
        current_step="Queued for processing",
        user_id=request.user_id,
        raw_video_s3_key=request.raw_video_s3_key,
    )

    background_tasks.add_task(
        run_processing_pipeline,
        request=request,
    )

    return {
        "success": True,
        "message": "Video processing started",
        "video_id": request.video_id,
        "estimated_time_minutes": "3-8 minutes",
        "status": "processing"
    }


async def run_processing_pipeline(
    request: ProcessVideoRequest,
):
    """Background task: run full AI pipeline"""
    try:
        log.info(
            f"Starting pipeline for {request.video_id}"
        )

        async def on_progress(percentage: int, message: str):
            store.save_status(
                video_id=request.video_id,
                status="processing",
                progress_percentage=percentage,
                current_step=message,
                user_id=request.user_id,
            )

        # CricketVideoProcessor.process() is declared async but is
        # actually synchronous, CPU/subprocess-bound work throughout
        # (YOLO inference, ffmpeg subprocess calls, boto3/shutil file
        # copies). Awaiting it directly on the main event loop would
        # block that loop for the whole run, starving every other
        # concurrent request -- including the client's own status
        # polling, which is exactly what caused polls to time out
        # mid-pipeline. Running it in a worker thread with its own
        # event loop keeps the main loop free to keep serving.
        result = await asyncio.to_thread(
            asyncio.run,
            processor.process(
                video_s3_key=request.raw_video_s3_key,
                video_id=request.video_id,
                user_id=request.user_id,
                match_id=request.match_id,
                progress_callback=on_progress,
            ),
        )

        highlights = result.get("highlights", [])
        total = len(highlights)
        status = result.get("status", "completed")

        store.save_highlights(request.video_id, highlights)
        store.save_status(
            video_id=request.video_id,
            status=status,
            progress_percentage=100,
            current_step=(
                "Complete!" if status == "completed" else "Failed"
            ),
            user_id=request.user_id,
            error=result.get("error"),
        )

        log.info(
            f"Processing done for {request.video_id}: "
            f"{total} highlights found"
        )

        # Send FCM notification
        if request.user_fcm_token and total > 0:
            await fcm_service.send_highlight_ready(
                fcm_token=request.user_fcm_token,
                video_id=request.video_id,
                highlights_count=total,
            )

        # Notify NestJS backend
        await webhook_service.notify_processing_complete(
            video_id=request.video_id,
            user_id=request.user_id,
            match_id=request.match_id,
            highlights=highlights,
            status=status,
        )

    except Exception as e:
        log.error(
            f"Pipeline failed for "
            f"{request.video_id}: {str(e)}"
        )
        store.save_status(
            video_id=request.video_id,
            status="failed",
            progress_percentage=100,
            current_step="Failed",
            user_id=request.user_id,
            error=str(e),
        )
        await webhook_service.notify_processing_failed(
            video_id=request.video_id,
            user_id=request.user_id,
            error=str(e)
        )


@router.get("/status/{video_id}")
async def get_processing_status(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get current processing status of a video"""
    status = store.get_status(video_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"No status found for video_id={video_id}",
        )
    return status


@router.post("/{video_id}/retry-ai")
async def retry_ai_processing(
    video_id: str,
    background_tasks: BackgroundTasks
):
    """Re-trigger AI processing if it failed"""
    return {
        "success": True,
        "message": "AI reprocessing started",
        "video_id": video_id
    }
