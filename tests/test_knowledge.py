"""Tests de la knowledge base (M6): los comentarios COMPLETAR no entran al prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import load_knowledge_base


def test_comentarios_html_no_entran_al_prompt():
    kb = load_knowledge_base()
    assert "<!--" not in kb
    assert "COMPLETAR" not in kb


def test_contenido_real_si_entra():
    kb = load_knowledge_base()
    # condiciones.md limpio sigue presente
    assert "Cambios y devoluciones" in kb
    assert "30 días" in kb
    # faq.md conserva las respuestas completas
    assert "mayoristas" in kb
