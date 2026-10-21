"""Connection pools and the permission-scoped session used for every user-facing read."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cached_property

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.auth.permissions import Permissions
from app.config import Settings


class Database:
    """Owns two pools: `app` (rag_app role, row-level security) and `admin` (ingestion)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.app_engine: AsyncEngine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )
        self.app_sessions = async_sessionmaker(self.app_engine, expire_on_commit=False)

    @cached_property
    def admin_engine(self) -> AsyncEngine:
        # Created lazily so API replicas that never ingest never open admin connections.
        return create_async_engine(
            self._settings.admin_database_url, pool_size=2, max_overflow=2, pool_pre_ping=True
        )

    @cached_property
    def admin_sessions(self) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(self.admin_engine, expire_on_commit=False)

    @asynccontextmanager
    async def authorized_session(self, permissions: Permissions) -> AsyncIterator[AsyncSession]:
        """A read-only transaction in which PostgreSQL row-level security sees the user's principals.

        `app.principals` is transaction-local (`set_config(..., true)`), so it can never
        leak to the next request that reuses the pooled connection.
        """
        async with self.app_sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await session.execute(
                text("SELECT set_config('app.principals', :principals, true)"),
                {"principals": json.dumps(sorted(permissions.principals))},
            )
            if self._settings.hnsw_iterative_scan:
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            yield session

    async def dispose(self) -> None:
        await self.app_engine.dispose()
        if "admin_engine" in self.__dict__:
            await self.admin_engine.dispose()
