"""Tests del manejo de token ML: refresh ante 401 y persistencia en kv_store."""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools


def _resp(status: int, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, json=payload or {}, request=httpx.Request("GET", "https://api.test")
    )


class _FakeClient:
    """Devuelve respuestas pre-armadas en orden; registra las llamadas."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.get_calls = 0
        self.post_calls = 0

    async def get(self, url, headers=None, params=None):
        self.get_calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def post(self, url, data=None):
        self.post_calls += 1
        return self.responses.pop(0)


def _patch_async_client(monkeypatch, fake):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(tools.httpx, "AsyncClient", _FakeAsyncClient)


def test_ml_get_json_refresca_token_ante_401(monkeypatch):
    fake = _FakeClient([_resp(401), _resp(200, {"ok": True})])
    _patch_async_client(monkeypatch, fake)

    refreshed = []

    async def fake_refresh():
        refreshed.append(True)
        return "token-nuevo"

    monkeypatch.setattr(tools, "_refresh_ml_token", fake_refresh)

    data = asyncio.run(tools._ml_get_json("https://api.mercadolibre.com/items/MLA1"))
    assert data == {"ok": True}
    assert refreshed == [True]
    assert fake.get_calls == 2


def test_refresh_ml_token_persiste_en_kv(monkeypatch):
    monkeypatch.setenv("ML_REFRESH_TOKEN", "ref-viejo")
    monkeypatch.setenv("ML_APP_ID", "app")
    monkeypatch.setenv("ML_CLIENT_SECRET", "secret")
    monkeypatch.setattr(tools, "_ml_token", "tok-viejo")

    fake = _FakeClient([_resp(200, {"access_token": "tok-nuevo", "refresh_token": "ref-nuevo"})])
    _patch_async_client(monkeypatch, fake)

    saved = {}

    async def fake_kv_set(key, value):
        saved[key] = value

    monkeypatch.setattr(tools, "kv_set", fake_kv_set)

    token = asyncio.run(tools._refresh_ml_token())
    assert token == "tok-nuevo"
    assert tools._ml_token == "tok-nuevo"
    assert saved == {"ml_access_token": "tok-nuevo", "ml_refresh_token": "ref-nuevo"}
    import os
    assert os.environ["ML_REFRESH_TOKEN"] == "ref-nuevo"


def test_load_ml_token_recupera_de_kv(monkeypatch):
    monkeypatch.setattr(tools, "_ml_token", "tok-del-env")

    async def fake_kv_get(key):
        return {"ml_access_token": "tok-persistido", "ml_refresh_token": "ref-persistido"}[key]

    monkeypatch.setattr(tools, "kv_get", fake_kv_get)

    asyncio.run(tools.load_ml_token())
    assert tools._ml_token == "tok-persistido"
    import os
    assert os.environ["ML_REFRESH_TOKEN"] == "ref-persistido"
