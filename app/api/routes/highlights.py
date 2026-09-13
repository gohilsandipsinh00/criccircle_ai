from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.local_store import store
from app.utils.logger import log

router = APIRouter(prefix="/api/v1/videos", tags=["highlights"])

VALID_ACTIONS = {"confirm", "reject", "edit"}


class FeedbackRequest(BaseModel):
    highlight_id: str
    action: str  # confirm | reject | edit
    corrected_type: Optional[str] = None
    corrected_start_seconds: Optional[float] = None
    corrected_end_seconds: Optional[float] = None
    quality_rating: Optional[int] = None  # 1-5
    notes: Optional[str] = None


@router.get("/{video_id}/highlights")
async def get_highlights(video_id: str):
    """Get all AI-detected highlights for a video"""
    highlights = store.get_highlights(video_id)
    return {
        "video_id": video_id,
        "highlights": highlights,
        "total": len(highlights),
    }


@router.post("/highlights/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    User confirms, rejects, or edits an AI-detected highlight.
    Called from the review screen. Persists both the raw feedback
    event (for stats/training-data export) and the highlight's own
    review status (so a later GET .../highlights reflects it).
    """
    if request.action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of {sorted(VALID_ACTIONS)}",
        )
    if request.quality_rating is not None and not (
        1 <= request.quality_rating <= 5
    ):
        raise HTTPException(
            status_code=400,
            detail="quality_rating must be between 1 and 5",
        )

    existing = store.find_highlight(request.highlight_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown highlight_id={request.highlight_id}",
        )

    review_status = {
        "confirm": "user_confirmed",
        "reject": "user_rejected",
        "edit": "user_edited",
    }[request.action]

    updated = store.update_highlight(
        request.highlight_id,
        review_status=review_status,
        corrected_type=request.corrected_type,
        corrected_start_seconds=request.corrected_start_seconds,
        corrected_end_seconds=request.corrected_end_seconds,
        quality_rating=request.quality_rating,
        notes=request.notes,
    )

    store.append_feedback({
        "highlight_id": request.highlight_id,
        "video_id": existing["video_id"],
        "action": request.action,
        "reviewed_type": existing.get("event_type"),
        "corrected_type": request.corrected_type,
        "corrected_start_seconds": request.corrected_start_seconds,
        "corrected_end_seconds": request.corrected_end_seconds,
        "quality_rating": request.quality_rating,
        "notes": request.notes,
    })

    log.info(
        f"Highlight {request.highlight_id}: action={request.action}"
    )

    return {
        "success": True,
        "highlight_id": request.highlight_id,
        "status": updated["review_status"],
    }


@router.get("/highlights/stats")
async def get_feedback_stats():
    """
    Session accuracy summary computed from feedback given so far.
    Not a claim about the model's true global accuracy -- just the
    honest ratio of what a human has confirmed vs. corrected.
    """
    return store.get_stats()
