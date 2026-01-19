from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Awaitable, TypeVar
import logging

import anyio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config.settings import get_settings
from app.db.migrations import run_migrations

DATABASE_URL = get_settings().database_url

logger = logging.getLogger(__name__)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_T = TypeVar("_T")


def _ensure_sqlite_dir(url: str) -> None:
    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return
    path = url[len(prefix) :]
    if path == ":memory:":
        return
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


async def _shielded(awaitable: Awaitable[_T]) -> _T:
    with anyio.CancelScope(shield=True):
        return await awaitable


async def _safe_rollback(session: AsyncSession) -> None:
    if not session.in_transaction():
        return
    try:
        await _shielded(session.rollback())
    except BaseException:
        return


async def _safe_close(session: AsyncSession) -> None:
    try:
        await _shielded(session.close())
    except BaseException:
        return


async def get_session() -> AsyncIterator[AsyncSession]:
    session = SessionLocal()
    try:
        yield session
    except BaseException:
        await _safe_rollback(session)
        raise
    finally:
        if session.in_transaction():
            await _safe_rollback(session)
        await _safe_close(session)


async def init_db() -> None:
    from app.db.models import Base

    _ensure_sqlite_dir(DATABASE_URL)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as session:
        try:
            updated = await run_migrations(session)
            if updated:
                logger.info("Applied database migrations count=%s", updated)
        except Exception:
            logger.exception("Failed to apply database migrations")
            if get_settings().database_migrations_fail_fast:
                raise
            
async def close_db() -> None:
    await engine.dispose()
