"""Tests del guardrail anti-alucinaciones (guardrails.validate_response, función pura)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails import validate_response

STOCK_CONTEXT = (
    "\n[Resultados verificados en Tienda Nube (tienda propia) para 'cocina']\n"
    "- Cocina de juguete Kitchen Fun | Precio: $45.500 | Stock: 3 unidades | "
    "https://klank.com.ar/productos/cocina-kitchen-fun\n"
    "[IMPORTANTE: Usá precios y links exactos.]"
)


def test_precio_correcto_pasa():
    ok, reason = validate_response(
        "Sí, tenemos la cocina Kitchen Fun a $45.500, quedan 3 unidades.",
        STOCK_CONTEXT,
        None,
    )
    assert ok, reason


def test_precio_inventado_falla():
    ok, reason = validate_response(
        "Sí, la cocina Kitchen Fun sale $39.999.",
        STOCK_CONTEXT,
        None,
    )
    assert not ok
    assert "precio no verificado" in reason


def test_url_de_resultados_pasa():
    ok, reason = validate_response(
        "Acá la tenés: https://klank.com.ar/productos/cocina-kitchen-fun",
        STOCK_CONTEXT,
        None,
    )
    assert ok, reason


def test_url_inventada_falla():
    ok, reason = validate_response(
        "Mirala acá: https://klank.com.ar/productos/cocina-magica-deluxe",
        STOCK_CONTEXT,
        None,
    )
    assert not ok
    assert "url no verificada" in reason


def test_respuesta_sin_datos_contexto_vacio_pasa():
    ok, reason = validate_response(
        "Hola, ¿cómo estás? ¿Qué producto estás buscando?",
        "",
        None,
    )
    assert ok, reason


def test_precio_con_contexto_vacio_falla():
    ok, reason = validate_response(
        "La cocina sale $45.500.",
        "",
        None,
    )
    assert not ok
    assert reason == "precio con contexto vacío"


def test_frase_de_derivacion_pasa():
    ok, reason = validate_response(
        "Para esta consulta comunicate con un asesor de Klank al +5493513047511. "
        "Te van a ayudar a la brevedad.",
        "",
        None,
    )
    assert ok, reason


def test_dominio_propio_desnudo_pasa():
    ok, reason = validate_response(
        "Podés ver todo el catálogo en https://klank.com.ar",
        STOCK_CONTEXT,
        None,
    )
    assert ok, reason


def test_precio_sin_separador_de_miles_matchea_formateado():
    """Regresión: '$12000' en el contexto debe permitir '$12.000' en la respuesta
    (antes la regex parseaba '$12000' como '$120' y rechazaba de más)."""
    ok, reason = validate_response(
        "Sí, la tenemos a $12.000.",
        "\n[Resultados]\n- Pelota | Precio: $12000 | Stock: 4 unidades | https://klank.com.ar/p/pelota",
        None,
    )
    assert ok, reason


def test_precio_con_decimales_sin_miles():
    ok, reason = validate_response(
        "Sale $8500,50.",
        "\n[Resultados]\n- Item | Precio: $8500,50 | https://klank.com.ar/p/item",
        None,
    )
    assert ok, reason


def test_precio_en_tool_result_pasa():
    # El precio no está en el texto del contexto pero sí en el tool_result crudo
    ok, reason = validate_response(
        "El envío te queda en $2.500.",
        STOCK_CONTEXT,
        {"shipping_cost": 2500.0},
    )
    assert ok, reason
