"""
Caché en memoria del catálogo completo de Tienda Nube + búsqueda local.

El caché se refresca cada CATALOG_TTL_S segundos y alimenta SOLO las
alternativas y recomendaciones: la búsqueda primaria de stock sigue siendo en
vivo (tools.search_products). Si el caché nunca cargó, esas features degradan
a "no hay alternativas" y el flujo actual queda intacto.

La búsqueda local tolera typos y acentos (normalización + difflib, stdlib puro).
"""

import asyncio
import difflib
import logging
import os
import re
import time
import unicodedata

import httpx

from config import TN_ACCESS_TOKEN, TN_STORE_ID, CATALOG_TTL_S
from tools import TIMEOUT, _get_with_retry, _parse_tn_products

logger = logging.getLogger(__name__)

_catalog: list[dict] = []
_loaded_at: float = 0.0
_refresh_lock = asyncio.Lock()

_PER_PAGE = 200
_MAX_PAGES = 50  # tope de seguridad (10.000 productos)

# Resumen del catálogo que se inyecta a la knowledge base del bot (M6).
# Sin precios a propósito: los precios solo pueden venir de los bloques
# verificados de la conversación (guardrail).
KNOWLEDGE_SUMMARY_PATH = os.path.join(
    os.path.dirname(__file__), "knowledge", "catalogo_resumen.md"
)

# ─── Refresco y acceso ────────────────────────────────────────────────────────


async def refresh_catalog(force: bool = False) -> int:
    """
    Descarga el catálogo completo de TN (paginado) y reemplaza el caché.
    Si la descarga falla a mitad de camino, conserva la copia anterior —
    el caché nunca queda vacío por un error transitorio.
    Retorna la cantidad de productos en el caché.
    """
    global _catalog, _loaded_at
    if not force and _catalog and (time.monotonic() - _loaded_at) < CATALOG_TTL_S:
        return len(_catalog)

    async with _refresh_lock:
        if not force and _catalog and (time.monotonic() - _loaded_at) < CATALOG_TTL_S:
            return len(_catalog)

        tn_token = os.getenv("TN_ACCESS_TOKEN", TN_ACCESS_TOKEN)
        tn_store = os.getenv("TN_STORE_ID", TN_STORE_ID)
        url = f"https://api.tiendanube.com/v1/{tn_store}/products"
        headers = {
            "Authentication": f"bearer {tn_token}",
            "User-Agent": "Klank-Agent/1.0",
        }

        products: list[dict] = []
        page = 1
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                while page <= _MAX_PAGES:
                    resp = await _get_with_retry(
                        client, url, headers=headers,
                        params={"per_page": _PER_PAGE, "page": page},
                    )
                    if resp.status_code == 404:
                        # TN devuelve 404 cuando la página excede el total
                        break
                    resp.raise_for_status()
                    data = resp.json()
                    if not isinstance(data, list) or not data:
                        break
                    products.extend(_parse_tn_products(data))
                    if len(data) < _PER_PAGE:
                        break
                    page += 1
        except Exception as e:
            logger.error(
                "refresh_catalog falló en página %s: %s — se conserva la copia anterior (%s productos)",
                page, e, len(_catalog),
            )
            return len(_catalog)

        if products:
            _catalog = products
            _loaded_at = time.monotonic()
            logger.info("Catálogo TN cacheado: %s productos (%s página/s)", len(products), page)
            _update_catalog_summary(products)
        else:
            logger.warning(
                "refresh_catalog devolvió 0 productos — se conserva la copia anterior (%s)",
                len(_catalog),
            )
        return len(_catalog)


def _build_catalog_summary(products: list[dict]) -> str:
    """Resumen de categorías del catálogo para la knowledge base (sin precios)."""
    from collections import Counter
    total = len(products)
    with_stock = sum(1 for p in products if (p.get("stock") or 0) > 0)
    cat_total: Counter = Counter()
    cat_stock: Counter = Counter()
    for p in products:
        for c in p.get("categories") or []:
            cat_total[c] += 1
            if (p.get("stock") or 0) > 0:
                cat_stock[c] += 1
    lines = [
        "# Catálogo de la tienda — resumen",
        "",
        "<!-- AUTOGENERADO por catalog.py en cada refresco del catálogo. NO editar a mano. -->",
        "",
        f"Nuestra tienda propia (Tienda Nube) tiene {total} productos publicados, "
        f"{with_stock} con stock en este momento.",
        "",
        "Categorías del catálogo (con stock / total):",
    ]
    for cat, count in cat_total.most_common(40):
        lines.append(f"- {cat}: {cat_stock.get(cat, 0)} con stock / {count} publicados")
    lines += [
        "",
        "IMPORTANTE: este resumen sirve solo para orientar sobre qué tipo de productos "
        "vendemos. Para nombres, precios y stock exactos SIEMPRE usá los bloques de "
        "resultados verificados de la conversación — nunca cites precios desde acá.",
    ]
    return "\n".join(lines) + "\n"


def _update_catalog_summary(products: list[dict]) -> None:
    """Escribe knowledge/catalogo_resumen.md y recarga la knowledge base cacheada."""
    try:
        with open(KNOWLEDGE_SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(_build_catalog_summary(products))
        # Import diferido: agent importa catalog a nivel módulo (evita el ciclo)
        from agent import load_knowledge_base
        load_knowledge_base()
        logger.info("catalogo_resumen.md regenerado y knowledge base recargada")
    except Exception as e:
        logger.warning("No se pudo actualizar catalogo_resumen.md: %s", e)


async def get_catalog() -> list[dict]:
    """
    Retorna el catálogo cacheado. Si venció el TTL (o nunca cargó), dispara el
    refresco en background y devuelve la copia actual — nunca agrega latencia
    al mensaje del cliente.
    """
    if not _catalog or (time.monotonic() - _loaded_at) >= CATALOG_TTL_S:
        asyncio.create_task(refresh_catalog())
    return _catalog


# ─── Búsqueda local ───────────────────────────────────────────────────────────

_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "para", "con", "sin", "y", "o",
    "un", "una", "unos", "unas", "en", "por", "que",
}


def _normalize(text: str) -> str:
    """minúsculas + sin acentos ('Pelóta' -> 'pelota')."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.split(r"[^a-z0-9]+", _normalize(text))
        if len(t) >= 2 and t not in _STOPWORDS
    ]


def _token_score(query_token: str, product_tokens: list[str]) -> float:
    """Mejor coincidencia de un token del query contra los tokens del producto."""
    best = 0.0
    for pt in product_tokens:
        if query_token == pt:
            return 1.0
        if query_token in pt or pt in query_token:
            best = max(best, 0.9)
        else:
            best = max(best, difflib.SequenceMatcher(None, query_token, pt).ratio())
    return best


def _score(query_tokens: list[str], product: dict) -> float:
    """Score 0-1: promedio del mejor match de cada token del query (typo-tolerante)."""
    haystack = product.get("title", "") + " " + " ".join(product.get("categories") or [])
    ptoks = _tokens(haystack)
    if not query_tokens or not ptoks:
        return 0.0
    return sum(_token_score(qt, ptoks) for qt in query_tokens) / len(query_tokens)


def search_local(query: str, products: list[dict], limit: int = 5) -> list[dict]:
    """
    Busca en el catálogo cacheado con tolerancia a typos y acentos.
    Ordena: con stock primero, después por score.
    """
    qtoks = _tokens(query)
    if not qtoks:
        return []
    scored = []
    for p in products:
        s = _score(qtoks, p)
        if s >= 0.72:
            scored.append((s, p))
    scored.sort(key=lambda t: (-((t[1].get("stock") or 0) > 0), -t[0]))
    return [p for _, p in scored[:limit]]


def find_alternatives(
    query: str,
    products: list[dict],
    exclude_permalinks: frozenset = frozenset(),
    limit: int = 3,
) -> list[dict]:
    """
    Productos alternativos a un query sin stock/resultados: SOLO con stock > 0,
    excluyendo los ya mostrados. Primero match relajado por tokens; si faltan,
    completa con productos de la misma categoría del mejor match global.
    """
    qtoks = _tokens(query)
    if not qtoks:
        return []
    in_stock = [
        p for p in products
        if (p.get("stock") or 0) > 0 and p.get("permalink") not in exclude_permalinks
    ]

    scored = [(s, p) for p in in_stock if (s := _score(qtoks, p)) >= 0.45]
    scored.sort(key=lambda t: -t[0])
    result = [p for _, p in scored[:limit]]

    if len(result) < limit:
        # Completar por categoría del producto más parecido del catálogo entero
        # (aunque ese esté sin stock: sirve para ubicar la categoría afín).
        best_score, best = 0.0, None
        for p in products:
            s = _score(qtoks, p)
            if s > best_score:
                best_score, best = s, p
        if best is not None and best_score >= 0.6:
            cats = set(best.get("categories") or [])
            if cats:
                seen = {p.get("permalink") for p in result}
                for p in in_stock:
                    if len(result) >= limit:
                        break
                    if p.get("permalink") in seen:
                        continue
                    if cats & set(p.get("categories") or []):
                        result.append(p)
                        seen.add(p.get("permalink"))
    return result


def filter_products(
    products: list[dict],
    keywords: str | list[str],
    price_max: float | None = None,
    limit: int = 3,
) -> list[dict]:
    """
    Filtra el catálogo para recomendaciones: SOLO stock > 0, match relajado por
    keywords (título + categorías) y precio máximo opcional.
    """
    if isinstance(keywords, (list, tuple)):
        keywords = " ".join(keywords)
    qtoks = _tokens(keywords)
    out = []
    for p in products:
        if (p.get("stock") or 0) <= 0:
            continue
        if price_max is not None:
            try:
                price = float(p.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if price <= 0 or price > price_max:
                continue
        s = _score(qtoks, p)
        if qtoks and s < 0.45:
            continue
        out.append((s, p))
    out.sort(key=lambda t: -t[0])
    return [p for _, p in out[:limit]]
