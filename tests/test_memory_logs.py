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


def test_log_interaction_campos_observabilidad(tmp_path, monkeypatch):
    db_path = _setup_sqlite(tmp_path, monkeypatch)

    asyncio.run(log_interaction(
        phone_number="5493511111111",
        user_message="tienen mochilas?",
        response_text="No encontré, pero tenemos estas alternativas.",
        tool_used="tienda_nube",
        tool_result={"products": [], "alternatives": [{"title": "Bolso"}]},
        search_query="mochila escolar",
        results_count=0,
        alternatives_count=1,
        tokens_in=1200,
        tokens_out=80,
        model="gpt-4o-mini",
        kb_gap=False,
    ))

    async def fetch():
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM agent_logs")
            return dict(await cursor.fetchone())

    row = asyncio.run(fetch())
    assert row["search_query"] == "mochila escolar"
    assert row["results_count"] == 0
    assert row["alternatives_count"] == 1
    assert row["tokens_in"] == 1200
    assert row["tokens_out"] == 80
    assert row["model"] == "gpt-4o-mini"
    assert row["kb_gap"] == 0


def test_migracion_sqlite_agrega_columnas_nuevas(tmp_path, monkeypatch):
    """Una base vieja (sin columnas de observabilidad) se migra sola en _sqlite_init."""
    db_path = str(tmp_path / "vieja.db")
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_USE_POSTGRES", False)

    async def crear_tabla_vieja():
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE agent_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number TEXT NOT NULL,
                    direction TEXT, user_message TEXT, tool_used TEXT,
                    tool_result TEXT, response_text TEXT, escalated INTEGER,
                    processing_ms INTEGER, error TEXT, prompt_version TEXT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            await db.commit()

    asyncio.run(crear_tabla_vieja())
    asyncio.run(memory._sqlite_init())

    async def columnas():
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info(agent_logs)")
            return {r[1] for r in await cursor.fetchall()}

    cols = asyncio.run(columnas())
    for col, _ in memory._AGENT_LOG_EXTRA_COLS:
        assert col in cols, col


def test_kv_set_y_get_roundtrip_sqlite(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)

    assert asyncio.run(memory.kv_get("ml_access_token")) is None
    asyncio.run(memory.kv_set("ml_access_token", "tok-abc"))
    assert asyncio.run(memory.kv_get("ml_access_token")) == "tok-abc"
    # Upsert: sobrescribe el valor existente
    asyncio.run(memory.kv_set("ml_access_token", "tok-def"))
    assert asyncio.run(memory.kv_get("ml_access_token")) == "tok-def"
