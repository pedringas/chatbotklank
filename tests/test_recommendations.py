"""Tests de recomendaciones por catálogo (M4): intent, criterios y bloque de contexto."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from agent import _RECO_INTENT_RE, _recommendations_block
from guardrails import validate_response

CATALOGO = [
    {"title": "Juego de mesa Ludo Klank", "price": 9500, "stock": 7,
     "permalink": "https://klank.com.ar/p/ludo", "categories": ["Juegos de mesa"]},
    {"title": "Rompecabezas didáctico encastre madera", "price": 7000, "stock": 3,
     "permalink": "https://klank.com.ar/p/encastre", "categories": ["Juguetes", "Didácticos"]},
    {"title": "Consola retro portátil", "price": 60000, "stock": 5,
     "permalink": "https://klank.com.ar/p/consola", "categories": ["Electrónica"]},
]


def _fake_openai(monkeypatch, content: str):
    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setattr(
        agent, "_openai",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )


def _fake_catalog(monkeypatch, products):
    async def fake_get_catalog():
        return products

    monkeypatch.setattr(agent, "get_catalog", fake_get_catalog)


# ─── Intent ───────────────────────────────────────────────────────────────────

def test_intent_matchea_frases_de_recomendacion():
    for frase in (
        "qué me recomendás?",
        "busco un regalo",
        "algo para una nena de 3 años",
        "tienen juegos baratos?",
        "algo económico hasta $10000",
    ):
        assert _RECO_INTENT_RE.search(frase), frase


def test_intent_no_matchea_consultas_normales():
    for frase in ("hola", "cuánto sale el ludo?", "dónde está mi pedido"):
        assert not _RECO_INTENT_RE.search(frase), frase


# ─── Bloque de recomendaciones ────────────────────────────────────────────────

def test_recomendaciones_arma_bloque_y_pasa_guardrail(monkeypatch):
    _fake_openai(monkeypatch, '{"keywords": "juego de mesa", "age": null, "price_max": 10000}')
    _fake_catalog(monkeypatch, CATALOGO)

    block, result = asyncio.run(_recommendations_block("algo barato para regalar", []))
    assert "[RECOMENDACIONES verificadas de nuestro catálogo]" in block
    assert "https://klank.com.ar/p/ludo" in block
    assert result["criteria"]["price_max"] == 10000
    assert all(float(p["price"]) <= 10000 for p in result["products"])

    respuesta = "Te recomiendo el Juego de mesa Ludo Klank a $9.500: https://klank.com.ar/p/ludo"
    ok, reason = validate_response(respuesta, block, result)
    assert ok, reason


def test_recomendaciones_por_edad_usa_keywords(monkeypatch):
    _fake_openai(monkeypatch, '{"keywords": "juguete didáctico encastre", "age": 3, "price_max": null}')
    _fake_catalog(monkeypatch, CATALOGO)

    block, result = asyncio.run(_recommendations_block("algo para una nena de 3 años", []))
    assert block
    assert result["products"][0]["permalink"] == "https://klank.com.ar/p/encastre"


def test_sin_criterios_no_hay_bloque(monkeypatch):
    _fake_openai(monkeypatch, '{"keywords": "", "age": null, "price_max": null}')
    _fake_catalog(monkeypatch, CATALOGO)

    block, result = asyncio.run(_recommendations_block("busco un regalo", []))
    assert block == "" and result is None


def test_catalogo_vacio_no_hay_bloque(monkeypatch):
    _fake_openai(monkeypatch, '{"keywords": "juego de mesa", "age": null, "price_max": null}')
    _fake_catalog(monkeypatch, [])

    block, result = asyncio.run(_recommendations_block("qué me recomendás", []))
    assert block == "" and result is None


def test_respuesta_json_invalida_degrada(monkeypatch):
    _fake_openai(monkeypatch, "no soy json")
    _fake_catalog(monkeypatch, CATALOGO)

    block, result = asyncio.run(_recommendations_block("qué me recomendás", []))
    assert block == "" and result is None
