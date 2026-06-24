"""
Historial de conversación por número de teléfono.
Usa PostgreSQL en producción (DATABASE_URL) y SQLite como fallback local.
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES = bool(DATABASE_URL)

# ─── PostgreSQL ────────────────────────────────────────────────────────────────

if _USE_POSTGRES:
    import asyncpg

    _pg_pool = None

    async def _get_pool():
        global _pg_pool
        if _pg_pool is None:
            _pg_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        return _pg_pool

    async def init_db() -> None:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          BIGSERIAL PRIMARY KEY,
                    phone_number TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_phone ON conversations(phone_number)"
            )
        logger.info("PostgreSQL inicializado correctamente")

    async def get_history(phone_number: str, limit: int = 10) -> list[dict]:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content FROM conversations
                WHERE phone_number = $1
                ORDER BY id DESC LIMIT $2
                """,
                phone_number, limit,
            )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def save_message(phone_number: str, role: str, content: str) -> None:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO conversations (phone_number, role, content) VALUES ($1, $2, $3)",
                phone_number, role, content,
            )

    async def clear_history(phone_number: str) -> None:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conversations WHERE phone_number = $1", phone_number
            )

# ─── SQLite (fallback local) ───────────────────────────────────────────────────

else:
    import aiosqlite

    DB_PATH = "conversations.db"

    async def init_db() -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    timestamp    TEXT NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_phone ON conversations(phone_number)"
            )
            await db.commit()
        logger.info("SQLite inicializado (modo local)")

    async def get_history(phone_number: str, limit: int = 10) -> list[dict]:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT role, content FROM conversations
                WHERE phone_number = ?
                ORDER BY id DESC LIMIT ?
                """,
                (phone_number, limit),
            )
            rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    async def save_message(phone_number: str, role: str, content: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO conversations (phone_number, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (phone_number, role, content, datetime.utcnow().isoformat()),
            )
            await db.commit()

    async def clear_history(phone_number: str) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "DELETE FROM conversations WHERE phone_number = ?", (phone_number,)
            )
            await db.commit()
