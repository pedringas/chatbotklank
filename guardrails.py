"""
Guardrail post-generación contra alucinaciones.

Valida que los precios y URLs que genera el LLM existan en los datos verificados
que le pasamos (stock_context / tool_result). Si la respuesta menciona un precio
o link que no está en el contexto, el agente reintenta una vez y si vuelve a
fallar deriva a un asesor humano (ver _generate_validated_response en agent.py).
"""
import re
from urllib.parse import urlparse

# Copia local de agent.HANDOFF_PHRASE — evita import circular con agent.py.
# Si cambia allá, actualizar acá.
HANDOFF_PHRASE = "comunicate con un asesor de Klank al"

# Dominios propios: permitidos como URL desnuda aunque no estén en el contexto
# (ej. "mirá el catálogo en https://klank.com.ar").
KB_DOMAINS = {"klank.com.ar"}

_URL_RE = re.compile(r"https?://[^\s]+")

# Precios en formato argentino: $1.500 / $ 1.500,50 / $1500
_PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\$\s?\d+")

# Números sueltos dentro de tool_result (ej. {'price': 1500.0}) — sin símbolo $
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_TRAILING_PUNCT = ".,;:!?)\"'»]"


def _extract_urls(text: str) -> list[str]:
    """Extrae URLs limpiando puntuación final pegada (fin de oración, paréntesis)."""
    return [u.rstrip(_TRAILING_PUNCT) for u in _URL_RE.findall(text)]


def _normalize_price(raw: str) -> float:
    """'$ 1.500,50' -> 1500.5 (formato argentino: '.' miles, ',' decimales)."""
    digits = raw.lstrip("$").strip().replace(".", "").replace(",", ".")
    return float(digits)


def _allowed_prices(stock_context: str, tool_result) -> set[float]:
    allowed = {_normalize_price(m) for m in _PRICE_RE.findall(stock_context or "")}
    if tool_result is not None:
        raw = str(tool_result)
        allowed.update(_normalize_price(m) for m in _PRICE_RE.findall(raw))
        # Los tool_result traen precios como números pelados ({'price': 1500.0})
        allowed.update(float(m) for m in _NUMBER_RE.findall(raw))
    return allowed


def validate_response(response: str, stock_context: str, tool_result=None) -> tuple[bool, str]:
    """
    Valida una respuesta generada contra los datos verificados.
    Retorna (True, "") si es segura, o (False, motivo) si contiene un precio
    o URL que no aparece en stock_context ni en tool_result.
    """
    # La derivación a humano siempre es segura (no afirma datos de productos)
    if HANDOFF_PHRASE in response:
        return True, ""

    known_text = (stock_context or "") + str(tool_result or "")

    # ── URLs ────────────────────────────────────────────────────────────────
    for url in _extract_urls(response):
        if url in known_text:
            continue
        parsed = urlparse(url)
        netloc = parsed.netloc.removeprefix("www.")
        if netloc in KB_DOMAINS and parsed.path in ("", "/"):
            continue
        return False, f"url no verificada: {url}"

    # ── Precios ─────────────────────────────────────────────────────────────
    response_prices = [_normalize_price(m) for m in _PRICE_RE.findall(response)]
    if response_prices:
        if not (stock_context or "").strip():
            return False, "precio con contexto vacío"
        allowed = _allowed_prices(stock_context, tool_result)
        for price in response_prices:
            if price not in allowed:
                return False, f"precio no verificado: ${price:g}"

    return True, ""
