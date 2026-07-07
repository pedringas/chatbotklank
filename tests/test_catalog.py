"""Tests del caché de catálogo TN y la búsqueda local (typos, alternativas, filtros)."""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import catalog
from catalog import search_local, find_alternatives, filter_products

FIXTURE = [
    {"title": "Pelota de fútbol N°5", "price": 12000, "stock": 4,
     "permalink": "https://klank.com.ar/p/pelota-futbol", "categories": ["Deportes", "Juguetes"]},
    {"title": "Pelota de básquet", "price": 15000, "stock": 0,
     "permalink": "https://klank.com.ar/p/pelota-basquet", "categories": ["Deportes"]},
    {"title": "Muñeca bebé con accesorios", "price": 18000, "stock": 2,
     "permalink": "https://klank.com.ar/p/muneca-bebe", "categories": ["Juguetes", "Muñecas"]},
    {"title": "Set de acuarelas x24", "price": 8000, "stock": 10,
     "permalink": "https://klank.com.ar/p/acuarelas-24", "categories": ["Papelería", "Arte"]},
    {"title": "Cocina de juguete Kitchen Fun", "price": 45500, "stock": 0,
     "permalink": "https://klank.com.ar/p/cocina-kitchen", "categories": ["Juguetes"]},
    {"title": "Juego de mesa Ludo Klank", "price": 9500, "stock": 7,
     "permalink": "https://klank.com.ar/p/ludo", "categories": ["Juguetes", "Juegos de mesa"]},
]


# ─── search_local ─────────────────────────────────────────────────────────────

def test_search_local_typo_matchea():
    result = search_local("pelotta futbol", FIXTURE)
    assert result and result[0]["permalink"] == "https://klank.com.ar/p/pelota-futbol"


def test_search_local_acentos_no_importan():
    result = search_local("muñeca bebe", FIXTURE)
    assert result and result[0]["permalink"] == "https://klank.com.ar/p/muneca-bebe"


def test_search_local_stock_primero():
    result = search_local("pelota", FIXTURE)
    assert len(result) == 2
    assert (result[0].get("stock") or 0) > 0  # la de fútbol (con stock) antes que básquet


def test_search_local_sin_match_devuelve_vacio():
    assert search_local("notebook gamer", FIXTURE) == []


# ─── find_alternatives ────────────────────────────────────────────────────────

def test_alternativas_solo_con_stock():
    result = find_alternatives("pelota", FIXTURE)
    assert result
    assert all((p.get("stock") or 0) > 0 for p in result)
    assert "https://klank.com.ar/p/pelota-basquet" not in [p["permalink"] for p in result]


def test_alternativas_excluye_ya_mostrados():
    result = find_alternatives(
        "pelota", FIXTURE,
        exclude_permalinks=frozenset({"https://klank.com.ar/p/pelota-futbol"}),
    )
    assert "https://klank.com.ar/p/pelota-futbol" not in [p["permalink"] for p in result]


def test_alternativas_completa_por_categoria():
    # "cocina de juguete" está sin stock → debe sugerir otros Juguetes con stock
    result = find_alternatives("cocina de juguete", FIXTURE)
    assert result
    assert all((p.get("stock") or 0) > 0 for p in result)
    cats = {c for p in result for c in p.get("categories", [])}
    assert "Juguetes" in cats


def test_alternativas_query_irrelevante_vacio():
    assert find_alternatives("heladera side by side", FIXTURE) == []


# ─── filter_products ──────────────────────────────────────────────────────────

def test_filter_por_precio_maximo():
    result = filter_products(FIXTURE, "juguete", price_max=10000)
    assert result
    assert all(float(p["price"]) <= 10000 for p in result)
    assert all((p.get("stock") or 0) > 0 for p in result)


def test_filter_keywords_lista():
    result = filter_products(FIXTURE, ["juegos", "mesa"])
    assert result and result[0]["permalink"] == "https://klank.com.ar/p/ludo"


# ─── refresh_catalog ──────────────────────────────────────────────────────────

class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def get(self, url, headers=None, params=None):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_client(monkeypatch, fake):
    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(catalog.httpx, "AsyncClient", _FakeAsyncClient)


def _resp(status, payload=None):
    return httpx.Response(status, json=payload, request=httpx.Request("GET", "https://api.test"))


_TN_RAW = [{
    "name": {"es": "Pelota de fútbol"},
    "canonical_url": "https://klank.com.ar/p/pelota-futbol",
    "variants": [{"price": "12000", "stock": 4, "sku": "PEL-1"}],
    "images": [], "categories": [{"name": {"es": "Deportes"}}],
}]


def test_refresh_carga_catalogo(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_catalog", [])
    monkeypatch.setattr(catalog, "_loaded_at", 0.0)
    monkeypatch.setattr(catalog, "KNOWLEDGE_SUMMARY_PATH", str(tmp_path / "catalogo_resumen.md"))
    _patch_client(monkeypatch, _FakeClient([_resp(200, _TN_RAW)]))

    count = asyncio.run(catalog.refresh_catalog(force=True))
    assert count == 1
    assert catalog._catalog[0]["title"] == "Pelota de fútbol"
    assert catalog._catalog[0]["categories"] == ["Deportes"]
    # El refresh regenera el resumen para la knowledge base
    resumen = (tmp_path / "catalogo_resumen.md").read_text(encoding="utf-8")
    assert "1 productos publicados" in resumen
    assert "Deportes" in resumen


def test_build_catalog_summary_sin_precios():
    resumen = catalog._build_catalog_summary(FIXTURE)
    assert "$" not in resumen  # los precios solo viven en los bloques verificados
    assert "Juguetes" in resumen
    assert f"{len(FIXTURE)} productos publicados" in resumen


def test_refresh_fallido_conserva_copia_anterior(monkeypatch):
    monkeypatch.setattr(catalog, "_catalog", list(FIXTURE))
    monkeypatch.setattr(catalog, "_loaded_at", 0.0)
    _patch_client(monkeypatch, _FakeClient([httpx.ReadTimeout("t1"), httpx.ReadTimeout("t2")]))

    count = asyncio.run(catalog.refresh_catalog(force=True))
    assert count == len(FIXTURE)
    assert catalog._catalog == FIXTURE  # no se vació


def test_refresh_404_en_primera_pagina_no_pisa(monkeypatch):
    monkeypatch.setattr(catalog, "_catalog", list(FIXTURE))
    monkeypatch.setattr(catalog, "_loaded_at", 0.0)
    _patch_client(monkeypatch, _FakeClient([_resp(404)]))

    count = asyncio.run(catalog.refresh_catalog(force=True))
    assert count == len(FIXTURE)
