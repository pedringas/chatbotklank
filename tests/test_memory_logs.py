"""Tests de agent_logs y kv_store en SQLite (antes el logging fallaba silencioso sin Postgres)."""
import asyncio
import json
import sys
from pathlib import Path

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory
from agent_logger import log_interaction


def _setup_sqlite(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_USE_POSTGRES", False)
    asyncio.run(memory._sqlite_init())
    return db_path


def test_log_interaction_persiste_en_sqlite(tmp_path, monkeypatch):
    db_path = _setup_sqlite(tmp_path, monkeypatch)

    asyncio.run(log_interaction(
        phone_number="5493511111111",
        user_message="tienen cocinas?",
        response_text="Sí, tenemos.",
        tool_used="tienda_nube",
        tool_result={"products": [{"title": "Cocina"}]},
        escalated=False,
        processing_ms=1234,
        error=None,
    ))

    async def fetch():
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM agent_logs")
            return [dict(r) for r in await cursor.fetchall()]

    rows = asyncio.run(fetch())
    assert len(rows) == 1
    row = rows[0]
    assert row["phone_number"] == "5493511111111"
    assert row["tool_used"] == "tienda_nube"
    assert json.loads(row["tool_result"])["products"][0]["title"] == "Cocina"
    assert row["escalated"] == 0
    assert row["processing_ms"] == 1234


def test_kv_set_y_get_roundtrip_sqlite(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)

    assert asyncio.run(memory.kv_get("ml_access_token")) is None
    asyncio.run(memory.kv_set("ml_access_token", "tok-abc"))
    assert asyncio.run(memory.kv_get("ml_access_token")) == "tok-abc"
    # Upsert: sobrescribe el valor existente
    asyncio.run(memory.kv_set("ml_access_token", "tok-def"))
    assert asyncio.run(memory.kv_get("ml_access_token")) == "tok-def"
