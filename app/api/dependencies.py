from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_webhook_secret(
    x_webhook_secret: str = Header(default=None)
) -> None:
    """Shared-secret check for inbound calls from the NestJS backend."""
    if x_webhook_secret != settings.nestjs_webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret"
        )
