import firebase_admin
from firebase_admin import credentials, messaging
import os
from app.config import settings
from app.utils.logger import log


class FCMService:
    def __init__(self):
        if not firebase_admin._apps:
            if os.path.exists(
                settings.firebase_credentials_path
            ):
                cred = credentials.Certificate(
                    settings.firebase_credentials_path
                )
                firebase_admin.initialize_app(cred)
                log.info("Firebase initialized")
            else:
                log.warning(
                    "Firebase credentials not found. "
                    "FCM notifications disabled."
                )

    async def send_highlight_ready(
        self,
        fcm_token: str,
        video_id: str,
        highlights_count: int,
    ):
        """Notify user that highlights are ready"""
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="Your Highlights Are Ready!",
                    body=(
                        f"We found {highlights_count} "
                        f"great moments in your video. "
                        f"Tap to review!"
                    ),
                ),
                data={
                    "type": "highlight_ready",
                    "video_id": video_id,
                    "count": str(highlights_count),
                    "screen": "highlight_review",
                    "click_action": (
                        "FLUTTER_NOTIFICATION_CLICK"
                    ),
                },
                token=fcm_token,
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        icon="ic_notification",
                        color="#F0A500",
                        sound="default",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            badge=1,
                            sound="default",
                        ),
                    ),
                ),
            )

            response = messaging.send(message)
            log.info(
                f"FCM sent successfully: {response}"
            )
            return response

        except Exception as e:
            log.error(f"FCM send failed: {str(e)}")

    async def send_processing_failed(
        self,
        fcm_token: str,
        video_id: str,
    ):
        """Notify user that processing failed"""
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title="Video Processing Failed",
                    body=(
                        "We couldn't process your video. "
                        "Tap to retry."
                    ),
                ),
                data={
                    "type": "processing_failed",
                    "video_id": video_id,
                    "screen": "media_gallery",
                },
                token=fcm_token,
            )
            messaging.send(message)
        except Exception as e:
            log.error(f"FCM failed notification error: {e}")
