# Klank WhatsApp Agent

Agente de WhatsApp para Klank — responde consultas de clientes automáticamente, consulta stock en tiempo real en MercadoLibre y Tienda Nube, y deriva a humanos via Chatwoot cuando no puede resolver.

## Stack

- **Servidor:** FastAPI + Uvicorn
- **LLM:** OpenAI GPT-4o mini
- **WhatsApp:** Meta Cloud API (webhooks)
- **Stock:** MercadoLibre API + Tienda Nube API (fallback)
- **Panel:** Chatwoot
- **Memoria:** SQLite con aiosqlite
- **Deploy:** Railway

---

## 1. Setup local

```bash
# Clonar y crear entorno virtual
git clone <repo>
cd klank-agent
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Mac/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 2. Variables de entorno

```bash
cp .env.example .env
# Abrir .env y completar todos los valores
```

### Dónde obtener cada variable

| Variable | Dónde conseguirla |
|---|---|
| `META_VERIFY_TOKEN` | Vos lo elegís — cualquier string secreto |
| `META_ACCESS_TOKEN` | Meta for Developers → WhatsApp → API Setup → Permanent Token |
| `META_PHONE_NUMBER_ID` | Meta for Developers → WhatsApp → API Setup → Phone Number ID |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `ML_ACCESS_TOKEN` | https://developers.mercadolibre.com → Mis apps → Token |
| `ML_SELLER_ID` | Tu perfil de vendedor ML → ver en URL o API `/users/me` |
| `TN_ACCESS_TOKEN` | https://partners.tiendanube.com → Apps → Token |
| `TN_STORE_ID` | Panel de Tienda Nube → URL de tu tienda (número) |
| `CHATWOOT_BASE_URL` | URL de tu instancia, ej: `https://app.chatwoot.com` |
| `CHATWOOT_API_TOKEN` | Chatwoot → Settings → API Access Tokens |
| `CHATWOOT_INBOX_ID` | Chatwoot → Settings → Inboxes → tu inbox de WhatsApp |

## 3. Correr localmente

```bash
uvicorn main:app --reload
```

El servidor queda en `http://localhost:8000`. Podés probar `/health` en el browser.

Para exponer el webhook local a internet (necesario para Meta), usá ngrok:

```bash
ngrok http 8000
# Copiá la URL https que te da ngrok
```

## 4. Deploy en Railway

1. Crear cuenta en https://railway.app
2. Nuevo proyecto → Deploy from GitHub repo
3. En Settings → Variables, agregar todas las variables de `.env`
4. Railway detecta `railway.toml` y ejecuta uvicorn automáticamente
5. Copiar la URL pública que Railway asigna (ej: `https://klank-agent.up.railway.app`)

## 5. Configurar webhook en Meta for Developers

1. Ir a https://developers.facebook.com → tu app → WhatsApp → Configuration
2. En Webhook URL poner: `https://tu-dominio.railway.app/webhook`
3. En Verify Token poner el valor de `META_VERIFY_TOKEN` de tu `.env`
4. Click en "Verify and Save"
5. Suscribirse al campo `messages`

## 6. Completar la base de conocimiento

Editá los archivos en `/knowledge/` con la información real de Klank:

- `formas_de_pago.md` — métodos de pago, recargos en cuotas
- `envios_y_retiro.md` — costos de envío, operadores, horarios de retiro
- `condiciones.md` — política de devoluciones y cambios
- `faq.md` — preguntas frecuentes

El agente usa estos archivos como fuente exclusiva de verdad para responder sobre pagos, envíos y condiciones. Sin estos datos, derivará al humano.
