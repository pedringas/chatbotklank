import json
import logging
from memory import _pg_get_pool

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


async def log_interaction(
    phone_number: str,
    user_message: str,
    response_text: str,
    tool_used: str = None,
    tool_result: dict = None,
    escalated: bool = False,
    processing_ms: int = None,
    error: str = None,
) -> None:
    try:
        pool = await _pg_get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_logs
                    (phone_number, direction, user_message, tool_used, tool_result,
                     response_text, escalated, processing_ms, error, prompt_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                phone_number,
                "inbound",
                user_message,
                tool_used,
                json.dumps(tool_result) if tool_result else None,
                response_text,
                escalated,
                processing_ms,
                error,
                PROMPT_VERSION,
            )
    except Exception as e:
        logger.error("agent_logger: failed to log interaction for %s: %s", phone_number, e)
