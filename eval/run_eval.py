"""
Sistema de evaluación automática del agente WhatsApp — Klank
Uso:
  python run_eval.py --mode synthetic           # evalúa los 30 casos predefinidos
  python run_eval.py --mode production          # evalúa filas reales de agent_logs
  python run_eval.py --mode synthetic --dry-run # verifica carga sin llamar al bot ni al judge
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import requests
from openai import OpenAI
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

BOT_URL = os.getenv("BOT_URL", "https://chatbotklank-production.up.railway.app")
DATABASE_PUBLIC_URL = os.getenv("DATABASE_PUBLIC_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JUDGE_MODEL = "gpt-4o"

EVAL_DIR = Path(__file__).parent
CASES_FILE = EVAL_DIR / "eval_cases.json"
JUDGE_PROMPT_FILE = EVAL_DIR / "judge_prompt.txt"
REPORTS_DIR = EVAL_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Número de teléfono ficticio para las pruebas sintéticas
TEST_PHONE = "5400000000000"

# ─── Judge ────────────────────────────────────────────────────────────────────

def call_judge(
    mensaje_cliente: str,
    respuesta_bot: str,
    tool_result,
    criterio_fallo: str,
) -> dict:
    judge_prompt = JUDGE_PROMPT_FILE.read_text(encoding="utf-8")
    user_content = json.dumps(
        {
            "mensaje_cliente": mensaje_cliente,
            "respuesta_bot": respuesta_bot,
            "tool_result": tool_result,
            "criterio_fallo": criterio_fallo,
        },
        ensure_ascii=False,
        indent=2,
    )
    client = OpenAI(api_key=OPENAI_API_KEY)
    msg = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        temperature=0,
        messages=[
            {"role": "user", "content": f"{judge_prompt}\n\n{user_content}"},
        ],
    )
    raw = msg.choices[0].message.content.strip()
    return json.loads(raw)


# ─── Bot caller (modo sintético) ──────────────────────────────────────────────

def call_bot_synthetic(caso: dict) -> str:
    """
    Llama al bot real via HTTP con el mensaje del cliente.
    El tool_result_simulado se incluye como contexto en el mensaje para guiar
    la búsqueda — en modo sintético el bot hace una búsqueda real o responde
    según su knowledge base.
    Nota: el bot no recibe el tool_result_simulado directamente; este campo
    se usa solo por el judge para evaluar si el bot alucinó datos no disponibles.
    """
    phone = TEST_PHONE
    text = caso["input"]

    # Limpiar historial del número de prueba antes de cada caso
    try:
        requests.post(
            f"{BOT_URL}/admin/clear-history",
            json={"phone": phone},
            timeout=5,
        )
    except Exception:
        pass  # endpoint opcional, ignorar si no existe

    resp = requests.post(
        f"{BOT_URL}/eval/message",
        json={"phone": phone, "message": text, "tool_result": caso.get("tool_result_simulado")},
        timeout=30,
    )
    resp.raise_for_status()
    bot_response = resp.json().get("response", "")

    # Segundo turno si existe (sin tool_result — simula respuesta sin contexto de búsqueda)
    if caso.get("turno_2"):
        resp2 = requests.post(
            f"{BOT_URL}/eval/message",
            json={"phone": phone, "message": caso["turno_2"], "tool_result": {}},
            timeout=30,
        )
        resp2.raise_for_status()
        bot_response = resp2.json().get("response", "")

    return bot_response


# ─── Modo sintético ───────────────────────────────────────────────────────────

def run_synthetic(dry_run: bool = False) -> list[dict]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    print(f"Cargados {len(cases)} casos de {CASES_FILE}")

    results = []
    for caso in cases:
        print(f"  [{caso['id']}] {caso['input'][:60]}...", end=" ", flush=True)

        if dry_run:
            results.append({
                **caso,
                "respuesta_bot": "[DRY RUN]",
                "score": None,
                "justificacion": "[DRY RUN]",
                "alucino": None,
                "escalado": None,
            })
            print("skip")
            continue

        try:
            respuesta_bot = call_bot_synthetic(caso)
            judgment = call_judge(
                mensaje_cliente=caso["turno_2"] or caso["input"],
                respuesta_bot=respuesta_bot,
                tool_result=caso["tool_result_simulado"],
                criterio_fallo=caso["criterio_fallo"],
            )
            escalado = "asesor de klank" in respuesta_bot.lower()
            results.append({
                **caso,
                "respuesta_bot": respuesta_bot,
                "score": judgment["score"],
                "justificacion": judgment["justificacion"],
                "alucino": judgment.get("alucino", False),
                "escalado": escalado,
            })
            print(f"score={judgment['score']}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                **caso,
                "respuesta_bot": f"ERROR: {e}",
                "score": 0,
                "justificacion": traceback.format_exc(),
                "alucino": None,
                "escalado": None,
            })

    return results


# ─── Modo producción ──────────────────────────────────────────────────────────

def run_production() -> list[dict]:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("ERROR: psycopg2-binary no está instalado. Corré: pip install psycopg2-binary")
        sys.exit(1)

    if not DATABASE_PUBLIC_URL:
        print("ERROR: DATABASE_PUBLIC_URL no está configurado en .env")
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_PUBLIC_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Verificar columnas de evaluación
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'agent_logs'
        AND column_name IN ('evaluated', 'judge_score', 'judge_note')
    """)
    existing = {r["column_name"] for r in cur.fetchall()}
    missing = {"evaluated", "judge_score", "judge_note"} - existing
    if missing:
        print("\nFaltan columnas en agent_logs. Ejecutá esto en Railway Console:")
        print("ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS evaluated TIMESTAMPTZ;")
        print("ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS judge_score INTEGER;")
        print("ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS judge_note TEXT;")
        sys.exit(1)

    cur.execute("""
        SELECT id, phone_number, user_message, response_text, tool_used,
               escalated, timestamp, processing_ms
        FROM agent_logs
        WHERE evaluated IS NULL
        ORDER BY timestamp DESC
        LIMIT 50
    """)
    rows = cur.fetchall()
    print(f"Evaluando {len(rows)} filas de agent_logs...")

    results = []
    for row in rows:
        print(f"  [id={row['id']}] {str(row['user_message'])[:60]}...", end=" ", flush=True)
        try:
            judgment = call_judge(
                mensaje_cliente=row["user_message"] or "",
                respuesta_bot=row["response_text"] or "",
                tool_result=None,
                criterio_fallo="El bot no debe inventar productos, precios ni condiciones. Debe escalar quejas y consultas complejas.",
            )
            cur.execute(
                "UPDATE agent_logs SET evaluated = NOW(), judge_score = %s, judge_note = %s WHERE id = %s",
                (judgment["score"], judgment["justificacion"], row["id"]),
            )
            conn.commit()
            results.append({
                "id": row["id"],
                "categoria": row.get("tool_used") or "general",
                "input": row["user_message"],
                "respuesta_bot": row["response_text"],
                "score": judgment["score"],
                "justificacion": judgment["justificacion"],
                "alucino": judgment.get("alucino", False),
                "escalado": row.get("escalated", False),
                "es_critico": False,
                "timestamp": str(row.get("timestamp", "")),
                "tool_used": row.get("tool_used"),
                "processing_ms": row.get("processing_ms"),
            })
            print(f"score={judgment['score']}")
        except Exception as e:
            print(f"ERROR: {e}")

    cur.close()
    conn.close()
    return results


# ─── Reporte HTML ─────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Eval Klank — {{ mode }} — {{ run_at }}</title>
<style>
  body { font-family: system-ui, sans-serif; font-size: 13px; margin: 20px; background: #f8fafc; color: #1e293b; }
  h1 { font-size: 18px; margin-bottom: 4px; }
  .meta { color: #64748b; font-size: 12px; margin-bottom: 20px; }
  .summary { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 20px; text-align: center; }
  .stat .val { font-size: 28px; font-weight: bold; }
  .stat .lbl { font-size: 11px; color: #64748b; }
  table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  th { background: #1e293b; color: white; padding: 10px 12px; text-align: left; font-size: 12px; }
  td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: top; font-size: 12px; }
  tr:hover td { background: #f8fafc; }
  .score-5 { background: #22c55e; color: white; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .score-4 { background: #86efac; color: #166534; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .score-3 { background: #fbbf24; color: #78350f; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .score-2 { background: #fb923c; color: white; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .score-1 { background: #ef4444; color: white; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .score-0 { background: #94a3b8; color: white; border-radius: 4px; padding: 2px 8px; font-weight: bold; }
  .critico { color: #dc2626; font-weight: bold; }
  .aluci-true { color: #dc2626; font-weight: bold; }
  .aluci-false { color: #16a34a; }
  .resp { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help; }
  .cat-badge { background: #e2e8f0; border-radius: 4px; padding: 1px 6px; font-size: 11px; }
</style>
</head>
<body>
<h1>Eval Klank Bot — modo {{ mode }}</h1>
<div class="meta">{{ run_at }} · {{ results|length }} casos evaluados</div>

<div class="summary">
  <div class="stat"><div class="val">{{ avg_score }}</div><div class="lbl">Score promedio</div></div>
  <div class="stat"><div class="val" style="color:#dc2626">{{ total_alucinaciones }}</div><div class="lbl">Alucinaciones</div></div>
  <div class="stat"><div class="val" style="color:#dc2626">{{ criticos_fallidos }}</div><div class="lbl">Críticos fallidos</div></div>
  {% for cat, avg in cat_scores.items() %}
  <div class="stat"><div class="val">{{ avg }}</div><div class="lbl">{{ cat }}</div></div>
  {% endfor %}
</div>

<table>
<thead>
<tr>
  {% if mode == 'synthetic' %}<th>ID</th>{% else %}<th>Timestamp</th>{% endif %}
  <th>Categoría</th>
  <th>Mensaje cliente</th>
  <th>Respuesta bot</th>
  {% if mode == 'production' %}<th>Tool</th>{% endif %}
  <th>Score</th>
  <th>Justificación</th>
  <th>Alucinó</th>
  <th>Escaló</th>
  {% if mode == 'synthetic' %}<th>Crítico</th>{% endif %}
</tr>
</thead>
<tbody>
{% for r in results %}
<tr>
  {% if mode == 'synthetic' %}
  <td>{{ r.id }}</td>
  {% else %}
  <td style="white-space:nowrap">{{ r.timestamp[:16] if r.timestamp else '' }}</td>
  {% endif %}
  <td><span class="cat-badge">{{ r.categoria }}</span></td>
  <td>{{ r.input }}</td>
  <td class="resp" title="{{ r.respuesta_bot }}">{{ r.respuesta_bot[:200] if r.respuesta_bot else '' }}</td>
  {% if mode == 'production' %}<td>{{ r.tool_used or '' }}</td>{% endif %}
  <td><span class="score-{{ r.score or 0 }}">{{ r.score if r.score is not none else 'ERR' }}</span></td>
  <td>{{ r.justificacion }}</td>
  <td class="{{ 'aluci-true' if r.alucino else 'aluci-false' }}">{{ '✓ SÍ' if r.alucino else '✗' }}</td>
  <td>{{ '✓' if r.escalado else '✗' }}</td>
  {% if mode == 'synthetic' %}
  <td>{{ '⚠️' if r.es_critico else '' }}</td>
  {% endif %}
</tr>
{% endfor %}
</tbody>
</table>
</body>
</html>"""


def generate_report(results: list[dict], mode: str, dry_run: bool = False) -> Path:
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = datetime.now().strftime(f"%Y%m%d_%H%M%S_{mode}{'_dryrun' if dry_run else ''}.html")
    out = REPORTS_DIR / fname

    scored = [r for r in results if r.get("score") is not None]
    avg_score = round(sum(r["score"] for r in scored) / len(scored), 2) if scored else "N/A"

    from collections import defaultdict
    cat_totals = defaultdict(list)
    for r in scored:
        cat_totals[r["categoria"]].append(r["score"])
    cat_scores = {k: round(sum(v) / len(v), 1) for k, v in cat_totals.items()}

    total_alucinaciones = sum(1 for r in results if r.get("alucino"))
    criticos_fallidos = sum(1 for r in results if r.get("es_critico") and (r.get("score") or 5) < 3)

    env = Environment()
    tmpl = env.from_string(REPORT_TEMPLATE)
    html = tmpl.render(
        mode=mode,
        run_at=run_at,
        results=results,
        avg_score=avg_score,
        cat_scores=cat_scores,
        total_alucinaciones=total_alucinaciones,
        criticos_fallidos=criticos_fallidos,
    )
    out.write_text(html, encoding="utf-8")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "production"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "synthetic":
        results = run_synthetic(dry_run=args.dry_run)
    else:
        if args.dry_run:
            print("--dry-run no aplica en modo production")
            sys.exit(1)
        results = run_production()

    report_path = generate_report(results, args.mode, dry_run=args.dry_run)
    print(f"\nReporte generado: {report_path}")

    # Abrir en browser si es posible
    try:
        import webbrowser
        webbrowser.open(report_path.as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
