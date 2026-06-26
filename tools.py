"""
Consultas de stock a MercadoLibre y Tienda Nube.
TN es la fuente primaria (tienda propia, mejor margen); ML es el fallback automático.
"""

import logging
import os
import httpx
from config import ML_ACCESS_TOKEN, ML_SELLER_ID, TN_ACCESS_TOKEN, TN_STORE_ID

logger = logging.getLogger(__name__)
TIMEOUT = 10  # segundos para todas las llamadas externas

# Token en memoria — se actualiza automáticamente cuando expira
_ml_token = ML_ACCESS_TOKEN


async def _refresh_ml_token() -> str:
    """
    Renueva el access token de ML usando el refresh token.
    Actualiza _ml_token en memoria y ML_REFRESH_TOKEN en el proceso.
    """
    global _ml_token
    refresh_token = os.getenv("ML_REFRESH_TOKEN", "")
    app_id = os.getenv("ML_APP_ID", "")
    client_secret = os.getenv("ML_CLIENT_SECRET", "")

    if not refresh_token:
        logger.error("No hay ML_REFRESH_TOKEN configurado — no se puede renovar el token")
        return _ml_token

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                "https://api.mercadolibre.com/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": app_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            _ml_token = data["access_token"]
            # Actualizar refresh token en memoria si ML devuelve uno nuevo
            if data.get("refresh_token"):
                os.environ["ML_REFRESH_TOKEN"] = data["refresh_token"]
            logger.info("Token de ML renovado correctamente")
            return _ml_token
    except Exception as e:
        logger.error("Error renovando token ML: %s", e)
        return _ml_token


def _ml_headers() -> dict:
    return {"Authorization": f"Bearer {_ml_token}"}


async def search_mercadolibre(query: str) -> dict:
    """
    Busca productos del vendedor en ML que coincidan con el query.
    Si el token expiró (401) lo renueva automáticamente y reintenta una vez.
    Si falla, llama a search_tiendanube como fallback.
    """
    url = f"https://api.mercadolibre.com/users/{ML_SELLER_ID}/items/search"
    params = {"q": query, "limit": 10}

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.get(url, headers=_ml_headers(), params=params)
                if resp.status_code == 401 and attempt == 0:
                    await _refresh_ml_token()
                    continue
                resp.raise_for_status()
                data = resp.json()

            results_raw = data.get("results", [])
            item_ids = [r["id"] for r in results_raw[:5] if isinstance(r, dict) and "id" in r]
            if not item_ids:
                return await search_tiendanube(query)

            products = []
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for item_id in item_ids[:3]:
                    detail = await _get_ml_item(client, item_id)
                    if detail:
                        products.append(detail)

            if not products:
                return await search_tiendanube(query)

            return {"source": "mercadolibre", "products": products}

        except Exception as e:
            logger.error("Error en MercadoLibre search: %s", e)
            return await search_tiendanube(query)

    return await search_tiendanube(query)


async def _get_ml_item(client: httpx.AsyncClient, item_id: str) -> dict | None:
    """Obtiene detalle de un ítem de ML (uso interno)."""
    try:
        resp = await client.get(
            f"https://api.mercadolibre.com/items/{item_id}", headers=_ml_headers()
        )
        resp.raise_for_status()
        d = resp.json()
        permalink = d.get("permalink", "")
        logger.info("ML item %s: '%s' | precio=%s | stock=%s | link=%s",
                    item_id, d.get("title", ""), d.get("price"), d.get("available_quantity"), permalink)
        return {
            "title": d.get("title", ""),
            "price": d.get("price"),
            "stock": d.get("available_quantity", 0),
            "permalink": permalink,
            "thumbnail": d.get("thumbnail", ""),
        }
    except Exception as e:
        logger.warning("No se pudo obtener ítem ML %s: %s", item_id, e)
        return None


def _parse_tn_products(data: list) -> list:
    """Convierte respuesta cruda de TN en lista de productos normalizados."""
    products = []
    for item in data:
        # Buscar la variante con stock primero, sino la primera
        variants = item.get("variants", [])
        variant = next((v for v in variants if (v.get("stock") or 0) > 0), variants[0] if variants else {})
        image = ""
        if item.get("images"):
            image = item["images"][0].get("src", "")
        # Usar precio real (price); solo usar promotional_price si es menor (descuento activo)
        price = variant.get("price")
        promo = variant.get("promotional_price")
        if promo and price and float(promo) < float(price):
            price = promo
        link = item.get("canonical_url") or item.get("permalink", "")
        sku = variant.get("sku", "")
        title = item.get("name", {}).get("es", "") or str(item.get("name", ""))
        logger.info("TN producto: '%s' | precio=%s | stock=%s", title, price, variant.get("stock"))
        products.append({
            "title": title,
            "price": price,
            "stock": variant.get("stock"),
            "permalink": link,
            "thumbnail": image,
            "sku": sku,
        })
    return products


async def _tn_search_raw(query: str, headers: dict, url: str) -> list:
    """Hace una búsqueda en TN y retorna la lista cruda."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=headers, params={"q": query, "per_page": 10})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error("Error en búsqueda TN '%s': %s", query, e)
        return []


async def search_tiendanube(query: str) -> dict:
    """
    Busca productos en Tienda Nube.
    Hace dos pasadas: por frase exacta y por SKU si aplica.
    Retorna hasta 3 productos ordenando los que tienen stock primero.
    """
    tn_token = os.getenv("TN_ACCESS_TOKEN", TN_ACCESS_TOKEN)
    tn_store = os.getenv("TN_STORE_ID", TN_STORE_ID)
    url = f"https://api.tiendanube.com/v1/{tn_store}/products"
    headers = {
        "Authentication": f"bearer {tn_token}",
        "User-Agent": "Klank-Agent/1.0",
    }

    try:
        # Búsqueda principal por nombre/query
        raw = await _tn_search_raw(query, headers, url)

        # Si la query parece un SKU (alfanumérico corto), buscar también por SKU
        if not raw and len(query) <= 20 and query.replace("-", "").replace("_", "").isalnum():
            raw = await _tn_search_raw(query, headers, url)

        if not raw:
            return {"source": "tiendanube", "products": []}

        all_products = _parse_tn_products(raw)

        # Ordenar: primero los que tienen stock > 0
        with_stock = [p for p in all_products if (p.get("stock") or 0) > 0]
        without_stock = [p for p in all_products if (p.get("stock") or 0) == 0]
        products = (with_stock + without_stock)[:3]

        return {"source": "tiendanube", "products": products}

    except Exception as e:
        logger.error("Error en Tienda Nube search: %s", e)
        return {
            "source": "error",
            "products": [],
            "error": "No pude consultar el stock en este momento.",
        }


async def get_product_by_id_ml(item_id: str) -> dict:
    """
    Obtiene detalle completo de un producto específico de MercadoLibre.
    Retorna título, precio, stock, permalink y thumbnail.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"https://api.mercadolibre.com/items/{item_id}", headers=_ml_headers()
            )
            resp.raise_for_status()
            d = resp.json()
        return {
            "title": d.get("title", ""),
            "price": d.get("price"),
            "stock": d.get("available_quantity", 0),
            "permalink": d.get("permalink", ""),
            "thumbnail": d.get("thumbnail", ""),
        }
    except Exception as e:
        logger.error("Error obteniendo ítem ML %s: %s", item_id, e)
        return {"error": f"No se pudo obtener el producto {item_id}."}


async def get_order_tiendanube(order_id: str) -> dict:
    """Obtiene el estado de un pedido de Tienda Nube por número de orden."""
    tn_token = os.getenv("TN_ACCESS_TOKEN", TN_ACCESS_TOKEN)
    tn_store = os.getenv("TN_STORE_ID", TN_STORE_ID)
    headers = {
        "Authentication": f"bearer {tn_token}",
        "User-Agent": "Klank-Agent/1.0",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"https://api.tiendanube.com/v1/{tn_store}/orders/{order_id}",
                headers=headers,
            )
            if resp.status_code == 404:
                return {"error": "Pedido no encontrado en Tienda Nube."}
            resp.raise_for_status()
            d = resp.json()

        status_map = {
            "open": "Abierto / En proceso",
            "closed": "Completado",
            "cancelled": "Cancelado",
        }
        payment_map = {
            "pending": "Pago pendiente",
            "authorized": "Pago autorizado",
            "paid": "Pagado",
            "voided": "Pago anulado",
            "refunded": "Reembolsado",
        }
        shipping = d.get("shipping_tracking_number") or d.get("shipping", {}).get("tracking_number")
        tracking_url = d.get("shipping_tracking_url") or d.get("shipping", {}).get("tracking_url")

        return {
            "source": "tiendanube",
            "order_id": d.get("number") or order_id,
            "status": status_map.get(d.get("status", ""), d.get("status", "")),
            "payment_status": payment_map.get(d.get("payment_status", ""), d.get("payment_status", "")),
            "tracking_number": shipping,
            "tracking_url": tracking_url,
            "created_at": d.get("created_at", ""),
            "updated_at": d.get("updated_at", ""),
        }
    except Exception as e:
        logger.error("Error obteniendo pedido TN %s: %s", order_id, e)
        return {"error": "No se pudo consultar el pedido en este momento."}


async def get_order_mercadolibre(order_id: str) -> dict:
    """Obtiene el estado de un pedido de MercadoLibre por ID."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(
                f"https://api.mercadolibre.com/orders/{order_id}",
                headers=_ml_headers(),
            )
            if resp.status_code == 401:
                await _refresh_ml_token()
                resp = await client.get(
                    f"https://api.mercadolibre.com/orders/{order_id}",
                    headers=_ml_headers(),
                )
            if resp.status_code == 404:
                return {"error": "Pedido no encontrado en MercadoLibre."}
            resp.raise_for_status()
            d = resp.json()

        shipment_id = d.get("shipping", {}).get("id")
        tracking_number = None
        tracking_url = None
        if shipment_id:
            try:
                sh = await client.get(
                    f"https://api.mercadolibre.com/shipments/{shipment_id}",
                    headers=_ml_headers(),
                )
                if sh.is_success:
                    sh_data = sh.json()
                    tracking_number = sh_data.get("tracking_number")
                    tracking_url = sh_data.get("tracking_url")
            except Exception:
                pass

        status_map = {
            "confirmed": "Confirmado",
            "payment_required": "Pago pendiente",
            "payment_in_process": "Pago en proceso",
            "paid": "Pagado",
            "partially_paid": "Pago parcial",
            "cancelled": "Cancelado",
        }

        return {
            "source": "mercadolibre",
            "order_id": order_id,
            "status": status_map.get(d.get("status", ""), d.get("status", "")),
            "tracking_number": tracking_number,
            "tracking_url": tracking_url,
            "date_created": d.get("date_created", ""),
        }
    except Exception as e:
        logger.error("Error obteniendo pedido ML %s: %s", order_id, e)
        return {"error": "No se pudo consultar el pedido en este momento."}


async def search_products(query: str) -> dict:
    """
    Busca productos con TN como fuente primaria y ML como fallback.
    TN primero porque es la tienda propia (mejor margen para Klank).
    """
    result = await search_tiendanube(query)
    if result.get("products"):
        return result
    return await search_mercadolibre(query)


# ─── Test básico ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def _test():
        print("Probando search_mercadolibre...")
        result = await search_mercadolibre("funko pop")
        print(result)

    asyncio.run(_test())
