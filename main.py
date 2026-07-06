"""
Servidor FastAPI. Maneja el webhook de Meta/WhatsApp y expone /health.
Siempre retorna 200 a Meta para evitar reintentos; errores internos se loguean.
"""

import hashlib
import hmac
import json
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
    META_APP_SECRET,
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
    if not META_APP_SECRET:
        logger.error(
            "META_APP_SECRET no está configurado — la verificación de firma del "
            "webhook está DESHABILITADA. Cualquiera con la URL puede inyectar "
            "mensajes falsos. TODO: configurar META_APP_SECRET en Railway y hacerlo obligatorio."
        )
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

def _verify_meta_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Verifica la firma HMAC-SHA256 que Meta envía en X-Hub-Signature-256.
    La firma se calcula sobre el body crudo (bytes), por eso NO se debe parsear
    y re-serializar antes de verificar. Comparación en tiempo constante.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    received = signature_header.split("=", 1)[1]
    expected = hmac.new(
        META_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(received, expected)


@app.post("/webhook")
async def receive_webhook(request: Request):
    # La firma se calcula sobre el body CRUDO — hay que leer los bytes antes de parsear.
    raw_body = await request.body()

    # Verificación de firma. Si META_APP_SECRET no está configurado, se saltea
    # (ya se logueó ERROR al arrancar) para no romper producción antes de setearlo.
    if META_APP_SECRET:
        signature = request.headers.get("X-Hub-Signature-256")
        if not _verify_meta_signature(raw_body, signature):
            client_ip = request.client.host if request.client else "desconocida"
            logger.warning("Webhook con firma inválida — IP de origen: %s", client_ip)
            return Response(status_code=403)

    # Meta espera siempre 200; los errores internos se loguean sin propagar
    try:
        body = json.loads(raw_body)
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
                msg_type = msg.get("type")
                # Evitar loop si el mensaje viene del propio bot
                if msg.get("from") == bot_number:
                    continue

                phone = msg["from"]

                # Audio — no podemos procesarlo
                if msg_type == "audio":
                    await send_whatsapp_message(phone, "No puedo escuchar audios. Por favor escribime tu consulta y te ayudo.")
                    continue

                # Imagen — extraer texto con visión
                if msg_type == "image":
                    image_data = msg.get("image", {})
                    caption = image_data.get("caption", "")
                    media_id = image_data.get("id", "")
                    if media_id:
                        try:
                            image_url = await _get_whatsapp_media_url(media_id)
                            text = await _describe_image(image_url, caption)
                        except Exception as e:
                            logger.error("Error procesando imagen: %s", e)
                            await send_whatsapp_message(phone, "No pude procesar la imagen. ¿Podés describirme qué buscás?")
                            continue
                    elif caption:
                        text = caption
                    else:
                        await send_whatsapp_message(phone, "Recibí una imagen pero no pude procesarla. ¿Podés escribirme qué buscás?")
                        continue

                # Solo texto e imágenes procesadas pasan de acá
                elif msg_type != "text":
                    continue
                else:
                    text = msg["text"]["body"]

                logger.info("Mensaje recibido de %s: %s", phone, text[:60])

                # Mandar "buscando..." solo si hay una búsqueda real de stock o URL, nunca en saludos
                from agent import _extract_ml_item_id, _extract_ml_product_name, _extract_klank_product_name
                from memory import get_history as _get_history
                _text_lower = text.lower().strip()
                _greeting_prefixes = ("hola", "buenas", "buen", "buenos", "hi ", "hello", "hey")
                _skip_words = {
                    "hola", "buenas", "buen dia", "buen día", "buenos dias", "buenos días",
                    "buenas tardes", "buenas noches", "hi", "hello", "hey",
                    "si", "sí", "no", "ok", "dale", "gracias", "perfecto", "genial",
                }
                _correction_starts = ("no,", "no ", "ese no", "eso no", "pero", "tampoco", "incorrecto")
                _history = await _get_history(phone, limit=1)
                _is_first_message = len(_history) == 0
                _is_skip = (
                    _is_first_message  # primer mensaje: siempre saludar primero sin "buscando"
                    or _text_lower in _skip_words
                    or any(_text_lower.startswith(g) for g in _greeting_prefixes)
                    or _text_lower.startswith(_correction_starts)
                    or "no es el precio" in _text_lower
                    or "precio es" in _text_lower
                    or "ese precio" in _text_lower
                )
                needs_search = not _is_skip and (
                    _is_product_query(text)
                    or _extract_ml_item_id(text) is not None
                    or _extract_ml_product_name(text) is not None
                    or _extract_klank_product_name(text) is not None
                )
                # Sin mensaje de espera — el bot responde directamente cuando termina la búsqueda

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


async def _get_whatsapp_media_url(media_id: str) -> str:
    """Obtiene la URL de descarga de un media de WhatsApp."""
    headers = {"Authorization": f"Bearer {META_ACCESS_TOKEN}"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://graph.facebook.com/v19.0/{media_id}", headers=headers
        )
        resp.raise_for_status()
        return resp.json()["url"]


async def _describe_image(image_url: str, caption: str) -> str:
    """Usa GPT-4o para extraer información de una imagen de producto."""
    from openai import AsyncOpenAI
    from config import OPENAI_API_KEY
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "El cliente envió una imagen de un producto. "
        "Describí brevemente qué producto se ve para que pueda buscarlo en una tienda. "
        "Devolvé solo el nombre del producto en 2-4 palabras, sin explicación."
    )
    if caption:
        prompt += f" El cliente también escribió: '{caption}'"
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        max_tokens=50,
    )
    return completion.choices[0].message.content.strip()


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
            if not resp.is_success:
                logger.error("WhatsApp %s — status=%s body=%s", phone_number, resp.status_code, resp.text[:500])
            resp.raise_for_status()
            logger.info("Mensaje enviado a %s", phone_number)
    except httpx.HTTPStatusError:
        pass  # ya logueado arriba
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


# ─── Webhook de Chatwoot (notificaciones entrantes) ──────────────────────────

@app.post("/notifications")
async def chatwoot_notifications(_: Request):
    return JSONResponse({"status": "ok"})


# ─── Endpoint de evaluación (solo para el script eval/run_eval.py) ───────────

@app.post("/eval/clear-history")
async def eval_clear_history(request: Request):
    """Limpia el historial de conversación de un número (usado entre casos del eval)."""
    from memory import clear_history
    body = await request.json()
    phone = body.get("phone", "eval_test")
    await clear_history(phone)
    return JSONResponse({"status": "ok"})


@app.post("/eval/message")
async def eval_message(request: Request):
    """
    Recibe un mensaje del script de evaluación y devuelve la respuesta del bot.
    Si se incluye tool_result, lo inyecta como contexto directamente (bypass de TN/ML).
    """
    body = await request.json()
    phone = body.get("phone", "eval_test")
    message = body.get("message", "")
    tool_result = body.get("tool_result")  # dict o None

    if not message:
        return JSONResponse({"error": "message requerido"}, status_code=400)

    # Formatear tool_result como stock_context si se provee
    # Distinguimos "clave ausente" (no bypass) de "clave presente pero null" (bypass sin resultados)
    stock_context_override = None
    if "tool_result" in body:
        if tool_result:
            # Formatear precios TN/ML con etiquetas claras para evitar inversiones
            if "precio_tn" in tool_result or "precio_ml" in tool_result:
                lines = []
                if "precio_tn" in tool_result:
                    lines.append(f"- Precio en tienda oficial: ${tool_result['precio_tn']}")
                if "precio_ml" in tool_result:
                    lines.append(f"- Precio en MercadoLibre: ${tool_result['precio_ml']}")
                for k, v in tool_result.items():
                    if k not in ("precio_tn", "precio_ml"):
                        lines.append(f"- {k}: {v}")
            else:
                lines = [f"- {k}: {v}" for k, v in tool_result.items()]
            stock_context_override = (
                "\n[Datos simulados para evaluación]\n"
                + "\n".join(lines)
                + "\n[IMPORTANTE: Usá solo estos datos. No inventes información adicional.]"
            )
        else:
            stock_context_override = (
                "\n[Búsqueda realizada: sin resultados para este producto]"
                "\n[IMPORTANTE: No inventes productos ni links. Decí honestamente que no encontraste ese producto.]"
            )

    response = await process_message(phone, message, stock_context_override=stock_context_override)
    return JSONResponse({"response": response})


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "environment": ENVIRONMENT}
