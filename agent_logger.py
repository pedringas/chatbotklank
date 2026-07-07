import json
import logging

import aiosqlite

import memory

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3"  # v3: fraseo en positivo al recomendar la opción más barata

_COLUMNS = (
    "phone_number", "direction", "user_message", "tool_used", "tool_result",
    "response_text", "escalated", "processing_ms", "error", "prompt_version",
    "search_query", "results_count", "alternatives_count",
    "tokens_in", "tokens_out", "model", "kb_gap",
)


async def log_interaction(
    phone_number: str,
    user_message: str,
    response_text: str,
    tool_used: str = None,
    tool_result: dict = None,
    escalated: bool = False,
    processing_ms: int = None,
    error: str = None,
    direction: str = "inbound",
    prompt_version: str = None,
    search_query: str = None,
    results_count: int = None,
    alternatives_count: int = None,
    tokens_in: int = None,
    tokens_out: int = None,
    model: str = None,
    kb_gap: bool = False,
) -> None:
    values = (
        phone_number,
        direction,
        user_message,
        tool_used,
        json.dumps(tool_result) if tool_result else None,
        response_text,
        escalated,
        processing_ms,
        error,
        prompt_version or PROMPT_VERSION,
        search_query,
        results_count,
        alternatives_count,
        tokens_in,
        tokens_out,
        model,
        kb_gap,
    )
    try:
        if memory.is_postgres():
            placeholders = ", ".join(f"${i}" for i in range(1, len(_COLUMNS) + 1))
            pool = await memory._pg_get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    f"INSERT INTO agent_logs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                    *values,
                )
        else:
            placeholders = ", ".join("?" for _ in _COLUMNS)
            sqlite_values = tuple(int(v) if isinstance(v, bool) else v for v in values)
            async with aiosqlite.connect(memory.DB_PATH) as db:
                await db.execute(
                    f"INSERT INTO agent_logs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                    sqlite_values,
                )
                await db.commit()
    except Exception as e:
        logger.error("agent_logger: failed to log interaction for %s: %s", phone_number, e)
