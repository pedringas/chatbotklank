"""
Integración con Chatwoot para registrar conversaciones y hacer handoff a humanos.
Todas las funciones son async con httpx.
"""

import logging
import httpx
from config import (
    CHATWOOT_BASE_URL,
    CHATWOOT_API_TOKEN,
    CHATWOOT_INBOX_ID,
    CHATWOOT_ACCOUNT_ID,
)

logger = logging.getLogger(__name__)
TIMEOUT = 10


def _headers() -> dict:
    return {
        "api_access_token": CHATWOOT_API_TOKEN,
        "Content-Type": "application/json",
    }


def _base() -> str:
    return f"{CHATWOOT_BASE_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}"


async def create_or_get_contact(phone_number: str) -> str | None:
    """
    Busca el contacto por número. Si no existe, lo crea.
    Retorna el contact_id o None si falla.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            search = await client.get(
                f"{_base()}/contacts/search",
                headers=_headers(),
                params={"q": phone_number, "include_contacts": True},
            )
            search.raise_for_status()
            payload = search.json()
            # La API puede devolver {"payload": {"contacts": [...]}} o {"payload": [...]}
            if isinstance(payload, dict):
                inner = payload.get("payload", {})
                results = inner.get("contacts", []) if isinstance(inner, dict) else (inner if isinstance(inner, list) else [])
            else:
                results = []
            if results:
                return str(results[0]["id"])

            create = await client.post(
                f"{_base()}/contacts",
                headers=_headers(),
                json={"phone_number": f"+{phone_number}", "name": phone_number},
            )
            create.raise_for_status()
            return str(create.json()["id"])

    except Exception as e:
        logger.error("Error en create_or_get_contact para %s: %s", phone_number, e)
        return None


async def get_or_create_conversation(contact_id: str, phone_number: str) -> str | None:
    """
    Busca una conversación abierta del contacto en el inbox configurado.
    Si no existe, crea una nueva. Evita duplicar conversaciones por mensaje.
    Retorna el conversation_id o None si falla.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Buscar conversaciones existentes del contacto
            resp = await client.get(
                f"{_base()}/contacts/{contact_id}/conversations",
                headers=_headers(),
            )
            resp.raise_for_status()
            conversations = resp.json().get("payload", [])

            # Reutilizar la primera conversación abierta o pendiente en el inbox correcto
            for conv in conversations:
                if (
                    str(conv.get("inbox_id")) == str(CHATWOOT_INBOX_ID)
                    and conv.get("status") in ("open", "pending")
                ):
                    return str(conv["id"])

            # No hay conversación activa — crear una nueva
            create = await client.post(
                f"{_base()}/conversations",
                headers=_headers(),
                json={
                    "contact_id": contact_id,
                    "inbox_id": CHATWOOT_INBOX_ID,
                    "additional_attributes": {"phone_number": phone_number},
                },
            )
            create.raise_for_status()
            return str(create.json()["id"])

    except Exception as e:
        logger.error("Error en get_or_create_conversation para contacto %s: %s", contact_id, e)
        return None


async def send_message_to_chatwoot(
    conversation_id: str, message: str, message_type: str = "outgoing"
) -> None:
    """
    Registra un mensaje en una conversación de Chatwoot.
    message_type: "incoming" para mensajes del cliente, "outgoing" para respuestas del agente.
    """
    url = f"{_base()}/conversations/{conversation_id}/messages"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"content": message, "message_type": message_type, "private": False},
            )
            resp.raise_for_status()
    except Exception as e:
        logger.error("Error enviando mensaje a Chatwoot (conv %s): %s", conversation_id, e)


async def flag_for_human(conversation_id: str) -> None:
    """
    Cambia el estado de la conversación a 'pending' y agrega una nota interna
    para que el equipo humano sepa que debe intervenir.
    """
    base_conv = f"{_base()}/conversations/{conversation_id}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            await client.patch(
                base_conv,
                headers=_headers(),
                json={"status": "pending"},
            )
            await client.post(
                f"{base_conv}/messages",
                headers=_headers(),
                json={
                    "content": "⚠️ El agente no pudo resolver esta consulta",
                    "message_type": "activity",
                    "private": True,
                },
            )
    except Exception as e:
        logger.error("Error en flag_for_human (conv %s): %s", conversation_id, e)
