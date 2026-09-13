"""create ai_videos, ai_highlights, ai_processing_jobs tables

Revision ID: 23c29f9e1ba4
Revises:
Create Date: 2026-08-07 18:43:19.331460

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '23c29f9e1ba4'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


video_status = postgresql.ENUM(
    "uploaded", "downloading", "processing", "extracting_frames",
    "detecting_objects", "classifying_events", "cutting_clips",
    "adding_branding", "uploading_results", "completed", "failed",
    name="videostatus",
)

highlight_type = postgresql.ENUM(
    "six", "four", "wicket", "catch", "celebration", "runout", "unknown",
    name="highlighttype",
)

highlight_status = postgresql.ENUM(
    "ai_detected", "user_confirmed", "user_edited", "user_rejected",
    "published",
    name="highlightstatus",
)

job_status = postgresql.ENUM(
    "queued", "running", "completed", "failed", "retrying",
    name="jobstatus",
)


def upgrade() -> None:
    bind = op.get_bind()
    video_status.create(bind, checkfirst=True)
    highlight_type.create(bind, checkfirst=True)
    highlight_status.create(bind, checkfirst=True)
    job_status.create(bind, checkfirst=True)

    op.create_table(
        "ai_videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("match_id", sa.String(), nullable=True),
        sa.Column("raw_video_s3_key", sa.String(), nullable=False),
        sa.Column("raw_video_url", sa.String(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("resolution", sa.String(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column(
            "status", video_status, nullable=False,
            server_default="uploaded",
        ),
        sa.Column(
            "progress_percentage", sa.Float(), server_default="0.0",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "total_highlights_detected", sa.Integer(), server_default="0",
        ),
        sa.Column(
            "processing_duration_seconds", sa.Float(), nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("video_metadata", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_ai_videos_user_id", "ai_videos", ["user_id"]
    )
    op.create_index(
        "ix_ai_videos_match_id", "ai_videos", ["match_id"]
    )

    op.create_table(
        "ai_highlights",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("match_id", sa.String(), nullable=True),
        sa.Column("highlight_type", highlight_type, nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column(
            "start_timestamp_seconds", sa.Float(), nullable=False,
        ),
        sa.Column("end_timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("peak_timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("detection_method", sa.String(), nullable=True),
        sa.Column("audio_spike_magnitude", sa.Float(), nullable=True),
        sa.Column("yolo_confidence", sa.Float(), nullable=True),
        sa.Column("classifier_confidence", sa.Float(), nullable=True),
        sa.Column("clip_s3_key", sa.String(), nullable=True),
        sa.Column("clip_url", sa.String(), nullable=True),
        sa.Column("branded_clip_url", sa.String(), nullable=True),
        sa.Column("thumbnail_url", sa.String(), nullable=True),
        sa.Column(
            "status", highlight_status, server_default="ai_detected",
        ),
        sa.Column("user_confirmed_at", sa.DateTime(), nullable=True),
        sa.Column("user_label", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_ai_highlights_video_id", "ai_highlights", ["video_id"]
    )
    op.create_index(
        "ix_ai_highlights_user_id", "ai_highlights", ["user_id"]
    )

    op.create_table(
        "ai_processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "status", job_status, nullable=False,
            server_default="queued",
        ),
        sa.Column("current_step", sa.String(), nullable=True),
        sa.Column("progress_percentage", sa.Float(), server_default="0.0"),
        sa.Column("attempt_count", sa.Integer(), server_default="1"),
        sa.Column("max_attempts", sa.Integer(), server_default="3"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_ai_processing_jobs_video_id", "ai_processing_jobs", ["video_id"]
    )
    op.create_index(
        "ix_ai_processing_jobs_user_id", "ai_processing_jobs", ["user_id"]
    )


def downgrade() -> None:
    op.drop_table("ai_processing_jobs")
    op.drop_table("ai_highlights")
    op.drop_table("ai_videos")

    bind = op.get_bind()
    job_status.drop(bind, checkfirst=True)
    highlight_status.drop(bind, checkfirst=True)
    highlight_type.drop(bind, checkfirst=True)
    video_status.drop(bind, checkfirst=True)
