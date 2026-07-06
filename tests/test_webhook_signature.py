"""
Verifica la firma HMAC del webhook de Meta (POST /webhook).
- Firma válida → 200
- Firma inválida → 403
- Sin META_APP_SECRET configurado → se saltea la verificación (200)

Se monkeypatchea main._handle_webhook a un no-op para no tocar DB/OpenAI/red,
y main.META_APP_SECRET con un secreto de prueba.
"""

import hashlib
import hmac
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402

TEST_SECRET = "test_app_secret_123"
BODY = {"entry": []}  # sin messages: _handle_webhook (si corriera) no haría nada


def _sign(raw: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


@pytest.fixture
def client(monkeypatch):
    async def _noop_handle(_body):
        return None

    monkeypatch.setattr(main, "_handle_webhook", _noop_handle)
    monkeypatch.setattr(main, "META_APP_SECRET", TEST_SECRET)
    # TestClient sin context manager para no disparar el lifespan (init_db).
    return TestClient(main.app)


def test_firma_valida_devuelve_200(client):
    raw = json.dumps(BODY).encode()
    resp = client.post(
        "/webhook",
        content=raw,
        headers={"X-Hub-Signature-256": _sign(raw, TEST_SECRET)},
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_firma_invalida_devuelve_403(client):
    raw = json.dumps(BODY).encode()
    resp = client.post(
        "/webhook",
        content=raw,
        headers={"X-Hub-Signature-256": _sign(raw, "secreto_equivocado")},
    )
    assert resp.status_code == 403


def test_sin_header_devuelve_403(client):
    raw = json.dumps(BODY).encode()
    resp = client.post("/webhook", content=raw)
    assert resp.status_code == 403


def test_firma_valida_sobre_body_crudo_no_reserializado(client):
    # Body con espacios no estándar: si el server re-serializara antes de firmar,
    # la verificación fallaría. Debe pasar porque firma sobre los bytes crudos.
    raw = b'{"entry":   [],  "extra": "  spaced  "}'
    resp = client.post(
        "/webhook",
        content=raw,
        headers={"X-Hub-Signature-256": _sign(raw, TEST_SECRET)},
    )
    assert resp.status_code == 200


def test_sin_secret_configurado_saltea_verificacion(monkeypatch):
    async def _noop_handle(_body):
        return None

    monkeypatch.setattr(main, "_handle_webhook", _noop_handle)
    monkeypatch.setattr(main, "META_APP_SECRET", "")
    c = TestClient(main.app)
    raw = json.dumps(BODY).encode()
    # Sin firma y sin secret configurado → no se bloquea
    resp = c.post("/webhook", content=raw)
    assert resp.status_code == 200
