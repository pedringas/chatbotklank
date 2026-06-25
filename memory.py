"""
Historial de conversación y perfil de cliente por número de teléfono.
Intenta PostgreSQL (DATABASE_URL) al arrancar; si falla, cae a SQLite automáticamente.
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES = False  # se actualiza en init_db si la conexión tiene éxito

# ─── SQLite helpers ───────────────────────────────────────────────────────────

import aiosqlite

DB_PATH = "conversations.db"


async def _sqlite_init() -> None:
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
        await db.execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                phone_number TEXT PRIMARY KEY,
                name         TEXT,
                preferences  TEXT,
                notes        TEXT,
                updated_at   TEXT NOT NULL
            )
        """)
        await db.commit()
    logger.info("SQLite inicializado (modo local)")


async def _sqlite_get_history(phone_number: str, limit: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT role, content FROM conversations WHERE phone_number = ? ORDER BY id DESC LIMIT ?",
            (phone_number, limit),
        )
        rows = await cursor.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def _sqlite_save(phone_number: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO conversations (phone_number, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (phone_number, role, content, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def _sqlite_clear(phone_number: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM conversations WHERE phone_number = ?", (phone_number,)
        )
        await db.commit()


async def _sqlite_get_profile(phone_number: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT name, preferences, notes FROM customer_profiles WHERE phone_number = ?",
            (phone_number,),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def _sqlite_save_profile(phone_number: str, name: str | None, preferences: str | None, notes: str | None) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO customer_profiles (phone_number, name, preferences, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(phone_number) DO UPDATE SET
                name = COALESCE(excluded.name, customer_profiles.name),
                preferences = excluded.preferences,
                notes = excluded.notes,
                updated_at = excluded.updated_at
        """, (phone_number, name, preferences, notes, datetime.utcnow().isoformat()))
        await db.commit()


# ─── PostgreSQL helpers ────────────────────────────────────────────────────────

_pg_pool = None


async def _pg_get_pool():
    global _pg_pool
    if _pg_pool is None:
        import asyncpg
        _pg_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _pg_pool


async def _pg_init() -> None:
    pool = await _pg_get_pool()
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                phone_number TEXT PRIMARY KEY,
                name         TEXT,
                preferences  TEXT,
                notes        TEXT,
                updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    logger.info("PostgreSQL inicializado correctamente")


async def _pg_get_history(phone_number: str, limit: int) -> list[dict]:
    pool = await _pg_get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM conversations WHERE phone_number = $1 ORDER BY id DESC LIMIT $2",
            phone_number, limit,
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def _pg_save(phone_number: str, role: str, content: str) -> None:
    pool = await _pg_get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (phone_number, role, content) VALUES ($1, $2, $3)",
            phone_number, role, content,
        )


async def _pg_clear(phone_number: str) -> None:
    pool = await _pg_get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM conversations WHERE phone_number = $1", phone_number
        )


async def _pg_get_profile(phone_number: str) -> dict | None:
    pool = await _pg_get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT name, preferences, notes FROM customer_profiles WHERE phone_number = $1",
            phone_number,
        )
    return dict(row) if row else None


async def _pg_save_profile(phone_number: str, name: str | None, preferences: str | None, notes: str | None) -> None:
    pool = await _pg_get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO customer_profiles (phone_number, name, preferences, notes, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (phone_number) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, customer_profiles.name),
                preferences = EXCLUDED.preferences,
                notes = EXCLUDED.notes,
                updated_at = NOW()
        """, phone_number, name, preferences, notes)


# ─── Interfaz pública ─────────────────────────────────────────────────────────

async def init_db() -> None:
    global _USE_POSTGRES
    if DATABASE_URL:
        try:
            await _pg_init()
            _USE_POSTGRES = True
            return
        except Exception as e:
            logger.warning("No se pudo conectar a PostgreSQL (%s) — usando SQLite", e)
    await _sqlite_init()


async def get_history(phone_number: str, limit: int = 20) -> list[dict]:
    if _USE_POSTGRES:
        try:
            return await _pg_get_history(phone_number, limit)
        except Exception as e:
            logger.error("Error leyendo historial de PG: %s", e)
    return await _sqlite_get_history(phone_number, limit)


async def save_message(phone_number: str, role: str, content: str) -> None:
    if _USE_POSTGRES:
        try:
            await _pg_save(phone_number, role, content)
            return
        except Exception as e:
            logger.error("Error guardando en PG: %s", e)
    await _sqlite_save(phone_number, role, content)


async def clear_history(phone_number: str) -> None:
    if _USE_POSTGRES:
        try:
            await _pg_clear(phone_number)
            return
        except Exception as e:
            logger.error("Error borrando historial en PG: %s", e)
    await _sqlite_clear(phone_number)


async def get_profile(phone_number: str) -> dict | None:
    if _USE_POSTGRES:
        try:
            return await _pg_get_profile(phone_number)
        except Exception as e:
            logger.error("Error leyendo perfil de PG: %s", e)
    return await _sqlite_get_profile(phone_number)


async def save_profile(phone_number: str, name: str | None, preferences: str | None, notes: str | None) -> None:
    if _USE_POSTGRES:
        try:
            await _pg_save_profile(phone_number, name, preferences, notes)
            return
        except Exception as e:
            logger.error("Error guardando perfil en PG: %s", e)
    await _sqlite_save_profile(phone_number, name, preferences, notes)
