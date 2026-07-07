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

# Columnas de observabilidad de agent_logs (M5). Se agregan con ALTER si la
# tabla ya existía (Railway/SQLite local viejo) y van en el CREATE para
# instalaciones nuevas. kb_gap y results_count=0 son las señales que dicen
# qué falta documentar en knowledge/ y dónde falla la búsqueda.
_AGENT_LOG_EXTRA_COLS = (
    ("search_query", "TEXT"),
    ("results_count", "INTEGER"),
    ("alternatives_count", "INTEGER"),
    ("tokens_in", "INTEGER"),
    ("tokens_out", "INTEGER"),
    ("model", "TEXT"),
    ("kb_gap", "BOOLEAN"),
)


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
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number   TEXT NOT NULL,
                direction      TEXT,
                user_message   TEXT,
                tool_used      TEXT,
                tool_result    TEXT,
                response_text  TEXT,
                escalated      INTEGER DEFAULT 0,
                processing_ms  INTEGER,
                error          TEXT,
                prompt_version TEXT,
                timestamp      TEXT NOT NULL DEFAULT (datetime('now')),
                evaluated      TEXT,
                judge_score    INTEGER,
                judge_note     TEXT
            )
        """)
        # Migración de columnas nuevas para bases locales creadas antes
        # (SQLite no soporta ADD COLUMN IF NOT EXISTS)
        cursor = await db.execute("PRAGMA table_info(agent_logs)")
        existing = {row[1] for row in await cursor.fetchall()}
        for col, typ in _AGENT_LOG_EXTRA_COLS:
            if col not in existing:
                await db.execute(f"ALTER TABLE agent_logs ADD COLUMN {col} {typ}")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
        # agent_logs versionada acá — antes se creaba a mano en la consola de
        # Railway, y en un entorno nuevo el logging fallaba silenciosamente.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id             BIGSERIAL PRIMARY KEY,
                phone_number   TEXT NOT NULL,
                direction      TEXT,
                user_message   TEXT,
                tool_used      TEXT,
                tool_result    TEXT,
                response_text  TEXT,
                escalated      BOOLEAN DEFAULT FALSE,
                processing_ms  INTEGER,
                error          TEXT,
                prompt_version TEXT,
                timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                evaluated      TIMESTAMPTZ,
                judge_score    INTEGER,
                judge_note     TEXT
            )
        """)
        # Columnas de eval y observabilidad sobre la tabla preexistente de Railway
        # (no-op si ya están)
        extra_cols = (
            ("evaluated", "TIMESTAMPTZ"), ("judge_score", "INTEGER"), ("judge_note", "TEXT"),
        ) + _AGENT_LOG_EXTRA_COLS
        for col, typ in extra_cols:
            await conn.execute(f"ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS {col} {typ}")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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

def is_postgres() -> bool:
    """True si init_db conectó a PostgreSQL (se resuelve al arrancar)."""
    return _USE_POSTGRES


async def kv_get(key: str) -> str | None:
    """Lee un valor persistente del kv_store (ej: tokens renovados de ML)."""
    if _USE_POSTGRES:
        try:
            pool = await _pg_get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow("SELECT value FROM kv_store WHERE key = $1", key)
            return row["value"] if row else None
        except Exception as e:
            logger.error("Error leyendo kv_store[%s] de PG: %s", key, e)
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = await cursor.fetchone()
    return row[0] if row else None


async def kv_set(key: str, value: str) -> None:
    """Guarda un valor persistente en el kv_store."""
    if _USE_POSTGRES:
        try:
            pool = await _pg_get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kv_store (key, value, updated_at) VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, key, value)
            return
        except Exception as e:
            logger.error("Error guardando kv_store[%s] en PG: %s", key, e)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, value, datetime.utcnow().isoformat()))
        await db.commit()


# ─── Takeover humano (WhatsApp Coexistence) ────────────────────────────────────
# Cuando el dueño responde manualmente desde la app de WhatsApp Business (evento
# smb_message_echoes), el bot debe callarse para ese contacto durante un tiempo.
# Se persiste en kv_store para sobrevivir reinicios del proceso.

_TAKEOVER_KEY_PREFIX = "takeover:"


async def set_human_takeover(phone_number: str, hours: float) -> None:
    """Marca que un humano está atendiendo a este contacto; el bot no responde hasta que expire."""
    from datetime import timedelta
    expiry = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
    await kv_set(f"{_TAKEOVER_KEY_PREFIX}{phone_number}", expiry)


async def is_human_active(phone_number: str) -> bool:
    """True si hay un takeover humano vigente (no vencido) para este contacto."""
    expiry_str = await kv_get(f"{_TAKEOVER_KEY_PREFIX}{phone_number}")
    if not expiry_str:
        return False
    try:
        expiry = datetime.fromisoformat(expiry_str)
    except ValueError:
        return False
    if datetime.utcnow() >= expiry:
        await clear_human_takeover(phone_number)
        return False
    return True


async def clear_human_takeover(phone_number: str) -> None:
    """Termina el takeover humano de este contacto (el bot vuelve a responder)."""
    if _USE_POSTGRES:
        try:
            pool = await _pg_get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM kv_store WHERE key = $1", f"{_TAKEOVER_KEY_PREFIX}{phone_number}"
                )
            return
        except Exception as e:
            logger.error("Error borrando takeover de %s en PG: %s", phone_number, e)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM kv_store WHERE key = ?", (f"{_TAKEOVER_KEY_PREFIX}{phone_number}",)
        )
        await db.commit()


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
