"""
Servidor FastAPI. Maneja el webhook de Meta/WhatsApp y expone /health.
Siempre retorna 200 a Meta para evitar reintentos; errores internos se loguean.
"""

import logging
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse

from config import (
    META_VERIFY_TOKEN,
    META_ACCESS_TOKEN,
    META_PHONE_NUMBER_ID,
    PORT,
    ENVIRONMENT,
)
from memory import init_db
from agent import process_message, needs_human_handoff, load_knowledge_base, _is_product_query
from chatwoot import (
    create_or_get_contact,
    get_or_create_conversation,
    send_message_to_chatwoot,
    flag_for_human,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHATSAPP_API_URL = (
    f"https://graph.facebook.com/v19.0/{META_PHONE_NUMBER_ID}/messages"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    load_knowledge_base()
    logger.info("Klank Agent iniciado correctamente en puerto %s", PORT)
    yield


app = FastAPI(lifespan=lifespan)


# ─── Verificación de webhook (GET) ────────────────────────────────────────────

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        logger.info("Webhook de Meta verificado correctamente")
        return PlainTextResponse(challenge)

    logger.warning("Intento de verificación fallido — token incorrecto")
    return Response(status_code=403)


# ─── Recepción de mensajes (POST) ─────────────────────────────────────────────

@app.post("/webhook")
async def receive_webhook(request: Request):
    # Meta espera siempre 200; los errores internos se loguean sin propagar
    try:
        body = await request.json()
        await _handle_webhook(body)
    except Exception as e:
        logger.error("Error procesando webhook: %s", e)
    return JSONResponse({"status": "ok"})


async def _handle_webhook(body: dict) -> None:
    entry = body.get("entry", [])
    for e in entry:
        for change in e.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])
            metadata = value.get("metadata", {})
            bot_number = metadata.get("phone_number_id", "")

            for msg in messages:
                # Solo mensajes de texto; ignorar status updates y otros tipos
                if msg.get("type") != "text":
                    continue
                # Evitar loop si el mensaje viene del propio bot
                if msg.get("from") == bot_number:
                    continue

                phone = msg["from"]
                text = msg["text"]["body"]

                logger.info("Mensaje recibido de %s: %s", phone, text[:60])

                # Mandar mensaje de "buscando..." solo si es consulta de producto
                if _is_product_query(text):
                    await send_whatsapp_message(phone, "Dejame buscar un momento 🔍")

                response = await process_message(phone, text)
                await send_whatsapp_message(phone, response)

                # Registrar en Chatwoot
                contact_id = await create_or_get_contact(phone)
                if contact_id:
                    conv_id = await get_or_create_conversation(contact_id, phone)
                    if conv_id:
                        await send_message_to_chatwoot(conv_id, text, "incoming")
                        await send_message_to_chatwoot(conv_id, response, "outgoing")
                        if needs_human_handoff(response):
                            await flag_for_human(conv_id)


async def send_whatsapp_message(phone_number: str, message: str) -> None:
    """Envía un mensaje de texto via Meta Cloud API."""
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message},
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(WHATSAPP_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info("Mensaje enviado a %s", phone_number)
    except Exception as e:
        logger.error("Error enviando mensaje WhatsApp a %s: %s", phone_number, e)


# ─── Diagnóstico de conexiones ────────────────────────────────────────────────

@app.get("/diagnostics")
async def diagnostics():
    """Verifica en tiempo real la conexión con ML y Tienda Nube."""
    from tools import search_products
    from config import ML_SELLER_ID, TN_STORE_ID

    results = {}

    # Test MercadoLibre
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.mercadolibre.com/users/{ML_SELLER_ID}",
                headers={"Authorization": f"Bearer {os.getenv('ML_ACCESS_TOKEN', '')}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                results["mercadolibre"] = {
                    "status": "✅ conectado",
                    "seller": data.get("nickname", ""),
                    "seller_id": ML_SELLER_ID,
                }
            else:
                results["mercadolibre"] = {"status": f"❌ error {resp.status_code}"}
    except Exception as e:
        results["mercadolibre"] = {"status": f"❌ {str(e)}"}

    # Test Tienda Nube
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.tiendanube.com/v1/{TN_STORE_ID}/store",
                headers={
                    "Authentication": f"bearer {os.getenv('TN_ACCESS_TOKEN', '')}",
                    "User-Agent": "Klank-Agent/1.0",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                results["tiendanube"] = {
                    "status": "✅ conectado",
                    "store": data.get("name", {}).get("es", ""),
                    "store_id": TN_STORE_ID,
                }
            else:
                results["tiendanube"] = {"status": f"❌ error {resp.status_code}"}
    except Exception as e:
        results["tiendanube"] = {"status": f"❌ {str(e)}"}

    # Test búsqueda real — llamadas directas para ver errores
    from tools import search_tiendanube as _tn, search_mercadolibre as _ml
    try:
        tn_result = await _tn("juguete")
        results["busqueda_tn"] = {
            "fuente": tn_result.get("source"),
            "productos": len(tn_result.get("products", [])),
            "error": tn_result.get("error"),
        }
    except Exception as e:
        results["busqueda_tn"] = {"error": str(e)}

    try:
        ml_result = await _ml("juguete")
        results["busqueda_ml"] = {
            "fuente": ml_result.get("source"),
            "productos": len(ml_result.get("products", [])),
            "error": ml_result.get("error"),
        }
    except Exception as e:
        results["busqueda_ml"] = {"error": str(e)}

    return results


# ─── OAuth callback MercadoLibre (solo para setup inicial) ───────────────────

@app.get("/auth/callback")
async def ml_oauth_callback(request: Request):
    """
    Recibe el code de ML tras el flujo OAuth y lo intercambia por el access token.
    Usar solo durante el setup — una vez que tenés el token, podés ignorar esta ruta.
    """
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "No se recibió code de ML"}, status_code=400)

    ml_app_id = os.getenv("ML_APP_ID", "")
    # Railway genera base_url como http:// internamente — forzamos https para que coincida con ML
    base = str(request.base_url).replace("http://", "https://").rstrip("/")
    redirect_uri = f"{base}/auth/callback"

    client_secret = os.getenv("ML_CLIENT_SECRET", "")
    if not client_secret:
        return JSONResponse({
            "error": "Falta ML_CLIENT_SECRET en .env",
            "code_recibido": code,
            "instruccion": "Agregá ML_CLIENT_SECRET al .env con el Secret de tu app ML"
        })

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.mercadolibre.com/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": ml_app_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            if not resp.is_success:
                return JSONResponse({
                    "error_http": resp.status_code,
                    "error_ml": resp.json(),
                    "debug": {
                        "ml_app_id": ml_app_id,
                        "redirect_uri_usado": redirect_uri,
                        "client_secret_cargado": bool(client_secret),
                        "code": code[:10] + "...",
                    }
                }, status_code=400)
            data = resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "✅ access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "expires_in": data.get("expires_in"),
        "instruccion": "Copiá el access_token y pegalo en ML_ACCESS_TOKEN de tu .env"
    })


# ─── OAuth callback Tienda Nube (solo para setup inicial) ────────────────────

@app.get("/tn/callback")
async def tn_oauth_callback(request: Request):
    """
    Recibe el code de Tienda Nube tras la instalación de la app y lo intercambia por el access token.
    """
    code = request.query_params.get("code")
    if not code:
        return JSONResponse({"error": "No se recibió code de Tienda Nube"}, status_code=400)

    tn_app_id = os.getenv("TN_APP_ID", "")
    tn_client_secret = os.getenv("TN_CLIENT_SECRET", "")

    if not tn_app_id or not tn_client_secret:
        return JSONResponse({
            "error": "Faltan TN_APP_ID o TN_CLIENT_SECRET en .env",
            "code_recibido": code,
        })

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://www.tiendanube.com/apps/authorize/token",
                data={
                    "client_id": tn_app_id,
                    "client_secret": tn_client_secret,
                    "grant_type": "authorization_code",
                    "code": code,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({
        "✅ TN_ACCESS_TOKEN": data.get("access_token"),
        "TN_USER_ID": data.get("user_id"),
        "instruccion": "Copiá TN_ACCESS_TOKEN y pegalo en el .env"
    })


# Endpoints de privacidad requeridos por Tienda Nube (GDPR)
@app.post("/tn/store-redact")
@app.post("/tn/customers-redact")
@app.post("/tn/customers-data")
async def tn_privacy(_: Request):
    return JSONResponse({"status": "ok"})


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "environment": ENVIRONMENT}
