"""
Historial de conversación por número de teléfono usando SQLite asíncrono.
La base de datos se crea automáticamente al iniciar si no existe.
"""

import aiosqlite
from datetime import datetime

DB_PATH = "conversations.db"


async def init_db() -> None:
    """Crea la tabla si no existe. Llamar al arrancar el servidor."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_phone ON conversations(phone_number)"
        )
        await db.commit()


async def get_history(phone_number: str, limit: int = 10) -> list[dict]:
    """Devuelve los últimos `limit` mensajes del número, en orden cronológico."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT role, content FROM conversations
            WHERE phone_number = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (phone_number, limit),
        )
        rows = await cursor.fetchall()
    # Invertir para tener orden cronológico ascendente
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def save_message(phone_number: str, role: str, content: str) -> None:
    """Guarda un mensaje en el historial."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO conversations (phone_number, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (phone_number, role, content, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def clear_history(phone_number: str) -> None:
    """Elimina todo el historial de un número."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM conversations WHERE phone_number = ?", (phone_number,)
        )
        await db.commit()
