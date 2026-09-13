from loguru import logger
import sys
from app.config import settings


def setup_logger():
    logger.remove()

    # Console logger
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if settings.debug else "INFO",
        colorize=True,
    )

    # File logger
    logger.add(
        "logs/criccircle_ai_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time} | {level} | {name}:{line} | {message}",
        level="INFO",
    )

    return logger


log = setup_logger()
