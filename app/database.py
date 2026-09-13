from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
)
from app.config import settings

# Sync engine (for Alembic migrations)
SYNC_DATABASE_URL = settings.database_url
engine = create_engine(SYNC_DATABASE_URL)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Async engine (for FastAPI routes)
ASYNC_DATABASE_URL = settings.database_url.replace(
    "postgresql://",
    "postgresql+asyncpg://"
)
async_engine = create_async_engine(
    ASYNC_DATABASE_URL, echo=settings.debug
)
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
