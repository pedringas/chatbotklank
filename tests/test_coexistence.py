"""
Tests de WhatsApp Coexistence: takeover humano (memory.py) y ruteo del webhook
por campo (main.py). COEXISTENCE_MODE está en False por defecto — estos tests
lo activan explícitamente vía monkeypatch para no depender de env vars.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory


def _setup_sqlite(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_USE_POSTGRES", False)
    asyncio.run(memory._sqlite_init())
    return db_path


# ─── memory.py: takeover humano ────────────────────────────────────────────────

def test_is_human_active_falso_sin_takeover(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)
    assert asyncio.run(memory.is_human_active("5493511111111")) is False


def test_set_human_takeover_activa_is_human_active(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)
    phone = "5493511111111"
    asyncio.run(memory.set_human_takeover(phone, hours=24))
    assert asyncio.run(memory.is_human_active(phone)) is True


def test_takeover_vencido_se_limpia_solo(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)
    phone = "5493511111111"
    # Horas negativas = expiry en el pasado, simula que ya venció
    asyncio.run(memory.set_human_takeover(phone, hours=-1))
    assert asyncio.run(memory.is_human_active(phone)) is False
    # clear_human_takeover ya se llamó internamente; confirmarlo vía kv_get
    assert asyncio.run(memory.kv_get(f"takeover:{phone}")) is None


def test_clear_human_takeover_manual(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)
    phone = "5493511111111"
    asyncio.run(memory.set_human_takeover(phone, hours=24))
    assert asyncio.run(memory.is_human_active(phone)) is True
    asyncio.run(memory.clear_human_takeover(phone))
    assert asyncio.run(memory.is_human_active(phone)) is False


def test_takeover_es_por_contacto(tmp_path, monkeypatch):
    _setup_sqlite(tmp_path, monkeypatch)
    asyncio.run(memory.set_human_takeover("5493511111111", hours=24))
    assert asyncio.run(memory.is_human_active("5493511111111")) is True
    assert asyncio.run(memory.is_human_active("5493512222222")) is False


# ─── agent.py: mensaje de derivación y trigger de takeover ─────────────────────

def test_handoff_message_default_es_numero_separado(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "COEXISTENCE_MODE", False)
    assert agent._handoff_message() == agent.HANDOFF_FULL
    assert agent.HANDOFF_PHONE in agent._handoff_message()


def test_handoff_message_coexistence_no_menciona_otro_numero(monkeypatch):
    import agent
    monkeypatch.setattr(agent, "COEXISTENCE_MODE", True)
    msg = agent._handoff_message()
    assert msg == agent.HANDOFF_COEXISTENCE
    assert agent.HANDOFF_PHONE not in msg


def test_needs_human_handoff_reconoce_ambas_frases():
    import agent
    assert agent.needs_human_handoff(agent.HANDOFF_FULL) is True
    assert agent.needs_human_handoff(agent.HANDOFF_COEXISTENCE) is True
    assert agent.needs_human_handoff("Hola, ¿cómo estás?") is False


def test_maybe_trigger_takeover_noop_si_coexistence_off(monkeypatch, tmp_path):
    import agent
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "COEXISTENCE_MODE", False)
    asyncio.run(agent._maybe_trigger_takeover("5493511111111"))
    assert asyncio.run(memory.is_human_active("5493511111111")) is False


def test_maybe_trigger_takeover_activa_si_coexistence_on(monkeypatch, tmp_path):
    import agent
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "COEXISTENCE_MODE", True)
    monkeypatch.setattr(agent, "HUMAN_TAKEOVER_HOURS", 24)
    phone = "5493511111111"
    asyncio.run(agent._maybe_trigger_takeover(phone))
    assert asyncio.run(memory.is_human_active(phone)) is True


# ─── main.py: ruteo de webhook por campo ───────────────────────────────────────

def test_handle_smb_message_echo_noop_si_coexistence_off(monkeypatch, tmp_path):
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", False)
    value = {"messages": [{"to": "5493511111111", "type": "text", "text": {"body": "hola"}}]}
    asyncio.run(main._handle_smb_message_echo(value))
    assert asyncio.run(memory.is_human_active("5493511111111")) is False


def test_handle_smb_message_echo_activa_takeover(monkeypatch, tmp_path):
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)
    monkeypatch.setattr(main, "HUMAN_TAKEOVER_HOURS", 24)
    phone = "5493511111111"
    value = {"messages": [{"to": phone, "type": "text", "text": {"body": "ya te ayudo"}}]}
    asyncio.run(main._handle_smb_message_echo(value))
    assert asyncio.run(memory.is_human_active(phone)) is True


def test_handle_smb_message_echo_fallback_a_contacts(monkeypatch, tmp_path):
    """Si el mensaje no trae 'to'/'recipient_id', usa contacts[0].wa_id."""
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)
    monkeypatch.setattr(main, "HUMAN_TAKEOVER_HOURS", 24)
    phone = "5493511111111"
    value = {
        "contacts": [{"wa_id": phone}],
        "messages": [{"type": "text", "text": {"body": "dale"}}],
    }
    asyncio.run(main._handle_smb_message_echo(value))
    assert asyncio.run(memory.is_human_active(phone)) is True


def test_handle_smb_message_echo_sin_destinatario_no_rompe(monkeypatch, tmp_path):
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)
    value = {"messages": [{"type": "text", "text": {"body": "sin destinatario"}}]}
    asyncio.run(main._handle_smb_message_echo(value))  # no debe lanzar excepción


def test_handle_webhook_ignora_smb_app_state_sync(monkeypatch, tmp_path):
    """El evento de sync de historial nunca debe generar respuesta del bot."""
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)

    called = {"process_message": False}

    async def fake_process_message(*args, **kwargs):
        called["process_message"] = True
        return "no debería llegar acá"

    monkeypatch.setattr(main, "process_message", fake_process_message)

    body = {
        "entry": [{"changes": [{
            "field": "smb_app_state_sync",
            "value": {"state_sync": [{"type": "contact", "contact": {"phone_number": "+5493511111111"}}]},
        }]}]
    }
    asyncio.run(main._handle_webhook(body))
    assert called["process_message"] is False


def test_handle_webhook_mensaje_normal_sigue_llamando_al_bot(monkeypatch, tmp_path):
    """Un mensaje entrante normal (campo 'messages', sin takeover activo) sigue
    generando respuesta automática — el flujo actual no debe romperse."""
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)

    called = {}

    async def fake_process_message(phone, text):
        called["phone"] = phone
        called["text"] = text
        return "respuesta del bot"

    async def fake_send(phone, message):
        called["sent"] = message

    monkeypatch.setattr(main, "process_message", fake_process_message)
    monkeypatch.setattr(main, "send_whatsapp_message", fake_send)
    monkeypatch.setattr(main, "create_or_get_contact", lambda *a, **k: asyncio.sleep(0, result=None))

    body = {
        "entry": [{"changes": [{
            "field": "messages",
            "value": {
                "messages": [{"from": "5493511111111", "type": "text", "text": {"body": "hola, tienen pelotas?"}}],
                "metadata": {"phone_number_id": "999"},
            },
        }]}]
    }
    asyncio.run(main._handle_webhook(body))
    assert called.get("phone") == "5493511111111"
    assert called.get("sent") == "respuesta del bot"


def test_handle_webhook_silencia_bot_con_takeover_activo(monkeypatch, tmp_path):
    """Con takeover humano activo, el mensaje del cliente NO dispara al bot."""
    import main
    _setup_sqlite(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "COEXISTENCE_MODE", True)
    phone = "5493511111111"
    asyncio.run(memory.set_human_takeover(phone, hours=24))

    called = {"process_message": False}

    async def fake_process_message(*a, **k):
        called["process_message"] = True
        return "no debería responder"

    monkeypatch.setattr(main, "process_message", fake_process_message)

    body = {
        "entry": [{"changes": [{
            "field": "messages",
            "value": {
                "messages": [{"from": phone, "type": "text", "text": {"body": "hola de nuevo"}}],
                "metadata": {"phone_number_id": "999"},
            },
        }]}]
    }
    asyncio.run(main._handle_webhook(body))
    assert called["process_message"] is False
