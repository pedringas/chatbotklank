"""Tests de las señales de observabilidad (M5): kb_gap y conteos derivados."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import _detect_kb_gap, _derive_counts, HANDOFF_FULL


def test_kb_gap_respuesta_sin_dato_sin_tool():
    assert _detect_kb_gap(None, "No tengo ese dato, ¿podés darme más detalles?")


def test_kb_gap_handoff_sin_tool():
    assert _detect_kb_gap(None, HANDOFF_FULL)


def test_kb_gap_falso_si_hubo_tool():
    # Derivaciones/faltas de dato con tool usada no son gaps de knowledge
    assert not _detect_kb_gap("tienda_nube", "No tengo ese dato")
    assert not _detect_kb_gap("escalation_deterministic", HANDOFF_FULL)


def test_kb_gap_falso_en_respuesta_normal():
    assert not _detect_kb_gap(None, "Hola, ¿cómo estás? ¿Qué producto buscás?")


def test_derive_counts_products_y_alternatives():
    rc, ac = _derive_counts({"products": [1, 2], "alternatives": [1]})
    assert (rc, ac) == (2, 1)


def test_derive_counts_busqueda_vacia():
    rc, ac = _derive_counts({"products": []})
    assert (rc, ac) == (0, None)


def test_derive_counts_tool_result_no_dict():
    assert _derive_counts(None) == (None, None)
    assert _derive_counts({"order_id": "123"}) == (None, None)
