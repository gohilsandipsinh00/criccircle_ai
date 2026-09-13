import httpx
from typing import List, Dict, Optional
from app.config import settings
from app.utils.logger import log


class WebhookService:
    """
    Sends results back to NestJS backend
    so it can update DB and notify Flutter app.
    """

    def __init__(self):
        self.backend_url = settings.nestjs_backend_url
        self.secret = settings.nestjs_webhook_secret

    async def notify_processing_complete(
        self,
        video_id: str,
        user_id: str,
        match_id: Optional[str],
        highlights: List[Dict],
        status: str
    ):
        """Tell NestJS backend processing is done"""
        payload = {
            "event": "ai_processing_complete",
            "video_id": video_id,
            "user_id": user_id,
            "match_id": match_id,
            "status": status,
            "total_highlights": len(highlights),
            "highlights": highlights,
        }

        await self._post_webhook(
            "/webhooks/ai/complete", payload
        )

    async def notify_processing_failed(
        self,
        video_id: str,
        user_id: str,
        error: str
    ):
        """Tell NestJS backend processing failed"""
        payload = {
            "event": "ai_processing_failed",
            "video_id": video_id,
            "user_id": user_id,
            "error": error,
        }

        await self._post_webhook(
            "/webhooks/ai/failed", payload
        )

    async def _post_webhook(
        self,
        path: str,
        payload: Dict
    ):
        url = f"{self.backend_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Secret": self.secret,
        }

        try:
            async with httpx.AsyncClient(
                timeout=30.0
            ) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                log.info(
                    f"Webhook {path}: "
                    f"status={response.status_code}"
                )
        except Exception as e:
            log.error(
                f"Webhook failed for {path}: {str(e)}"
            )
