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

            item_ids = [r["id"] for r in data.get("results", [])[:5]]
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
        return {
            "title": d.get("title", ""),
            "price": d.get("price"),
            "stock": d.get("available_quantity", 0),
            "permalink": d.get("permalink", ""),
            "thumbnail": d.get("thumbnail", ""),
        }
    except Exception as e:
        logger.warning("No se pudo obtener ítem ML %s: %s", item_id, e)
        return None


async def search_tiendanube(query: str) -> dict:
    """
    Busca productos en Tienda Nube que coincidan con el query.
    Retorna lista de hasta 3 productos con nombre, precio, stock, link e imagen.
    """
    # Leer token dinámicamente para capturar actualizaciones sin reiniciar
    tn_token = os.getenv("TN_ACCESS_TOKEN", TN_ACCESS_TOKEN)
    tn_store = os.getenv("TN_STORE_ID", TN_STORE_ID)
    url = f"https://api.tiendanube.com/v1/{tn_store}/products"
    headers = {
        "Authentication": f"bearer {tn_token}",
        "User-Agent": "Klank-Agent/1.0",
    }
    params = {"q": query, "fields": "name,variants,permalink,images,canonical_url", "per_page": 5}

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, list):
            logger.warning("Tienda Nube devolvió respuesta inesperada: %s", type(data))
            return {"source": "tiendanube", "products": []}

        products = []
        for item in data[:3]:
            variant = item.get("variants", [{}])[0] if item.get("variants") else {}
            image = ""
            if item.get("images"):
                image = item["images"][0].get("src", "")
            # Usar precio promocional si existe, sino precio normal
            price = variant.get("promotional_price") or variant.get("price")
            # Preferir canonical_url (tienda propia) sobre permalink
            link = item.get("canonical_url") or item.get("permalink", "")
            products.append(
                {
                    "title": item.get("name", {}).get("es", "") or str(item.get("name", "")),
                    "price": price,
                    "stock": variant.get("stock"),
                    "permalink": link,
                    "thumbnail": image,
                }
            )

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
