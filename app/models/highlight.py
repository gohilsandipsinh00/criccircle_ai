from sqlalchemy import (
    Column, String, Float,
    DateTime, Enum, Text
)
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum
from app.database import Base


class HighlightType(str, enum.Enum):
    SIX = "six"
    FOUR = "four"
    WICKET = "wicket"
    CATCH = "catch"
    CELEBRATION = "celebration"
    RUNOUT = "runout"
    UNKNOWN = "unknown"


class HighlightStatus(str, enum.Enum):
    AI_DETECTED = "ai_detected"
    USER_CONFIRMED = "user_confirmed"
    USER_EDITED = "user_edited"
    USER_REJECTED = "user_rejected"
    PUBLISHED = "published"


class Highlight(Base):
    __tablename__ = "ai_highlights"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    video_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    match_id = Column(String, nullable=True)

    # AI Detection results
    highlight_type = Column(
        Enum(HighlightType),
        nullable=False
    )
    confidence_score = Column(Float, nullable=False)

    # Timestamps in original video
    start_timestamp_seconds = Column(Float, nullable=False)
    end_timestamp_seconds = Column(Float, nullable=False)
    peak_timestamp_seconds = Column(Float, nullable=False)

    # Detection details
    detection_method = Column(String, nullable=True)
    # audio_spike / yolo_detection / classifier / combined

    audio_spike_magnitude = Column(Float, nullable=True)
    yolo_confidence = Column(Float, nullable=True)
    classifier_confidence = Column(Float, nullable=True)

    # Output files (S3 keys)
    clip_s3_key = Column(String, nullable=True)
    clip_url = Column(String, nullable=True)
    branded_clip_url = Column(String, nullable=True)
    thumbnail_url = Column(String, nullable=True)

    # User interaction
    status = Column(
        Enum(HighlightStatus),
        default=HighlightStatus.AI_DETECTED
    )
    user_confirmed_at = Column(DateTime, nullable=True)
    user_label = Column(String, nullable=True)

    # Metadata
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
