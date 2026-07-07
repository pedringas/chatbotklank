"""Tests de _get_with_retry: reintenta timeouts y 5xx, no reintenta 4xx."""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import _get_with_retry


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://api.test"))


class _FakeClient:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls = 0

    async def get(self, url, headers=None, params=None):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_reintenta_timeout_y_recupera():
    fake = _FakeClient([httpx.ReadTimeout("timeout"), _resp(200)])
    resp = asyncio.run(_get_with_retry(fake, "https://api.test", backoff=0))
    assert resp.status_code == 200
    assert fake.calls == 2


def test_reintenta_5xx_y_recupera():
    fake = _FakeClient([_resp(503), _resp(200)])
    resp = asyncio.run(_get_with_retry(fake, "https://api.test", backoff=0))
    assert resp.status_code == 200
    assert fake.calls == 2


def test_no_reintenta_4xx():
    fake = _FakeClient([_resp(404), _resp(200)])
    resp = asyncio.run(_get_with_retry(fake, "https://api.test", backoff=0))
    assert resp.status_code == 404
    assert fake.calls == 1


def test_agota_reintentos_y_propaga_timeout():
    fake = _FakeClient([httpx.ReadTimeout("t1"), httpx.ReadTimeout("t2")])
    try:
        asyncio.run(_get_with_retry(fake, "https://api.test", backoff=0))
        assert False, "debería haber propagado el timeout"
    except httpx.ReadTimeout:
        pass
    assert fake.calls == 2


def test_ultimo_intento_5xx_se_devuelve():
    fake = _FakeClient([_resp(500), _resp(502)])
    resp = asyncio.run(_get_with_retry(fake, "https://api.test", backoff=0))
    assert resp.status_code == 502
    assert fake.calls == 2
