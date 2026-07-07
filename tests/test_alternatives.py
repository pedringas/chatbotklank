"""Tests del bloque de alternativas (M3): formato, guardrail y degradación sin caché."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import _alternatives_block, _format_alternatives_block, format_stock_context
from guardrails import validate_response

ALTS = [
    {"title": "Pelota de fútbol N°5", "price": 12000, "stock": 4,
     "permalink": "https://klank.com.ar/p/pelota-futbol", "categories": ["Deportes"]},
    {"title": "Juego de mesa Ludo", "price": 9500, "stock": 7,
     "permalink": "https://klank.com.ar/p/ludo", "categories": ["Juguetes"]},
]


def test_bloque_alternativas_pasa_el_guardrail():
    """El requisito de diseño clave: lo ofrecido desde el bloque no dispara el guardrail."""
    stock_context = (
        "\n[Búsqueda realizada: sin resultados para este producto en ninguna fuente]"
        + _format_alternatives_block(ALTS)
    )
    respuesta = (
        "Ese producto no lo tenemos ahora, pero sí tenemos la Pelota de fútbol N°5 a $12.000: "
        "https://klank.com.ar/p/pelota-futbol"
    )
    ok, reason = validate_response(respuesta, stock_context, None)
    assert ok, reason


def test_precio_fuera_del_bloque_sigue_bloqueado():
    stock_context = (
        "\n[Búsqueda realizada: sin resultados para este producto en ninguna fuente]"
        + _format_alternatives_block(ALTS)
    )
    ok, reason = validate_response("Tenemos una pelota a $99.999", stock_context, None)
    assert not ok
    assert "precio no verificado" in reason


def test_alternatives_block_usa_el_catalogo(monkeypatch):
    async def fake_get_catalog():
        return ALTS

    monkeypatch.setattr(agent, "get_catalog", fake_get_catalog)
    block, alts = asyncio.run(_alternatives_block("pelota futbol"))
    assert "[ALTERNATIVAS verificadas con stock en nuestra tienda]" in block
    assert "https://klank.com.ar/p/pelota-futbol" in block
    assert alts and alts[0]["title"] == "Pelota de fútbol N°5"


def test_alternatives_block_excluye_permalinks(monkeypatch):
    async def fake_get_catalog():
        return ALTS

    monkeypatch.setattr(agent, "get_catalog", fake_get_catalog)
    block, alts = asyncio.run(_alternatives_block(
        "pelota futbol", frozenset({"https://klank.com.ar/p/pelota-futbol"})
    ))
    assert "pelota-futbol" not in block


def test_cache_vacio_degrada_sin_bloque(monkeypatch):
    """Sin catálogo cargado, el contexto queda idéntico al flujo anterior."""
    async def fake_get_catalog():
        return []

    monkeypatch.setattr(agent, "get_catalog", fake_get_catalog)
    block, alts = asyncio.run(_alternatives_block("pelota"))
    assert block == "" and alts == []


def test_error_de_catalogo_degrada_sin_bloque(monkeypatch):
    async def fake_get_catalog():
        raise RuntimeError("catálogo caído")

    monkeypatch.setattr(agent, "get_catalog", fake_get_catalog)
    block, alts = asyncio.run(_alternatives_block("pelota"))
    assert block == "" and alts == []


def test_query_vacio_degrada_sin_bloque():
    block, alts = asyncio.run(_alternatives_block(""))
    assert block == "" and alts == []


def test_format_stock_context_renderiza_products_y_alternatives():
    ctx = format_stock_context({
        "products": [{"title": "Cocina Kitchen Fun", "price": 45500, "stock": 0,
                      "permalink": "https://klank.com.ar/p/cocina"}],
        "alternatives": ALTS,
    })
    assert "[Resultados verificados en nuestra tienda]" in ctx
    assert "SIN STOCK" in ctx
    assert "[ALTERNATIVAS verificadas con stock en nuestra tienda]" in ctx
    assert "https://klank.com.ar/p/ludo" in ctx


def test_format_stock_context_products_vacio_con_alternatives():
    ctx = format_stock_context({"products": [], "alternatives": ALTS})
    assert "sin resultados" in ctx
    assert "[ALTERNATIVAS verificadas con stock en nuestra tienda]" in ctx


def test_format_stock_context_dict_simple_sigue_igual():
    """Regresión: el formato legacy del eval no cambia."""
    ctx = format_stock_context({"producto": "Set cocina", "precio": 8500})
    assert "[Resultados verificados en nuestra tienda]" in ctx
    assert "precio: $8500" in ctx
