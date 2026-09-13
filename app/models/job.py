from sqlalchemy import (
    Column, String, Float,
    DateTime, Enum, Text, Integer
)
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from app.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ProcessingJob(Base):
    """Tracks a single AI processing run for a video."""
    __tablename__ = "ai_processing_jobs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    video_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    status = Column(
        Enum(JobStatus),
        default=JobStatus.QUEUED,
        nullable=False
    )
    current_step = Column(String, nullable=True)
    progress_percentage = Column(Float, default=0.0)

    attempt_count = Column(Integer, default=1)
    max_attempts = Column(Integer, default=3)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
