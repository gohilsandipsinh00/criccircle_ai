import redis
from typing import Optional
from app.config import settings
from app.utils.logger import log


class QueueService:
    """
    Thin wrapper around Redis used to track in-flight processing
    jobs and enforce MAX_CONCURRENT_JOBS, mirroring the Bull queue
    used on the NestJS side.
    """

    ACTIVE_JOBS_KEY = "criccircle:ai:active_jobs"

    def __init__(self):
        self.redis = redis.from_url(
            settings.redis_url, decode_responses=True
        )
        self.max_concurrent = settings.max_concurrent_jobs

    def can_accept_job(self) -> bool:
        active = self.redis.scard(self.ACTIVE_JOBS_KEY)
        return active < self.max_concurrent

    def register_job(self, video_id: str) -> None:
        self.redis.sadd(self.ACTIVE_JOBS_KEY, video_id)
        log.info(f"Job registered: {video_id}")

    def release_job(self, video_id: str) -> None:
        self.redis.srem(self.ACTIVE_JOBS_KEY, video_id)
        log.info(f"Job released: {video_id}")

    def set_progress(
        self, video_id: str, percentage: int, message: str
    ) -> None:
        self.redis.hset(
            f"criccircle:ai:progress:{video_id}",
            mapping={"percentage": percentage, "message": message},
        )
        self.redis.expire(
            f"criccircle:ai:progress:{video_id}", 3600
        )

    def get_progress(self, video_id: str) -> Optional[dict]:
        data = self.redis.hgetall(
            f"criccircle:ai:progress:{video_id}"
        )
        return data or None
