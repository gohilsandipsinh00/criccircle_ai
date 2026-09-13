from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Enum, Text, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from app.database import Base


class VideoStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    EXTRACTING_FRAMES = "extracting_frames"
    DETECTING_OBJECTS = "detecting_objects"
    CLASSIFYING_EVENTS = "classifying_events"
    CUTTING_CLIPS = "cutting_clips"
    ADDING_BRANDING = "adding_branding"
    UPLOADING_RESULTS = "uploading_results"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "ai_videos"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id = Column(String, nullable=False, index=True)
    match_id = Column(String, nullable=True, index=True)

    # S3 locations
    raw_video_s3_key = Column(String, nullable=False)
    raw_video_url = Column(String, nullable=False)

    # Video metadata
    duration_seconds = Column(Float, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    resolution = Column(String, nullable=True)
    fps = Column(Float, nullable=True)

    # Processing status
    status = Column(
        Enum(VideoStatus),
        default=VideoStatus.UPLOADED,
        nullable=False
    )
    progress_percentage = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # Results
    total_highlights_detected = Column(Integer, default=0)
    processing_duration_seconds = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Extra data
    video_metadata = Column(JSON, nullable=True)
