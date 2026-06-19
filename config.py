"""
Carga y valida todas las variables de entorno requeridas.
Si alguna falta, el servidor no arranca e indica exactamente cuál falta.
"""

import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = [
    "META_VERIFY_TOKEN",
    "META_ACCESS_TOKEN",
    "META_PHONE_NUMBER_ID",
    "OPENAI_API_KEY",
    "ML_ACCESS_TOKEN",
    "ML_SELLER_ID",
    "TN_ACCESS_TOKEN",
    "TN_STORE_ID",
    "CHATWOOT_BASE_URL",
    "CHATWOOT_API_TOKEN",
    "CHATWOOT_INBOX_ID",
    "CHATWOOT_ACCOUNT_ID",
]


def _load() -> dict:
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Variables de entorno faltantes: {', '.join(missing)}\n"
            "Copiá .env.example a .env y completá los valores."
        )
    return {v: os.environ[v] for v in REQUIRED_VARS}


_cfg = _load()

META_VERIFY_TOKEN: str = _cfg["META_VERIFY_TOKEN"]
META_ACCESS_TOKEN: str = _cfg["META_ACCESS_TOKEN"]
META_PHONE_NUMBER_ID: str = _cfg["META_PHONE_NUMBER_ID"]

OPENAI_API_KEY: str = _cfg["OPENAI_API_KEY"]

ML_ACCESS_TOKEN: str = _cfg["ML_ACCESS_TOKEN"]
ML_SELLER_ID: str = _cfg["ML_SELLER_ID"]

TN_ACCESS_TOKEN: str = _cfg["TN_ACCESS_TOKEN"]
TN_STORE_ID: str = _cfg["TN_STORE_ID"]

CHATWOOT_BASE_URL: str = _cfg["CHATWOOT_BASE_URL"].rstrip("/")
CHATWOOT_API_TOKEN: str = _cfg["CHATWOOT_API_TOKEN"]
CHATWOOT_INBOX_ID: str = _cfg["CHATWOOT_INBOX_ID"]
CHATWOOT_ACCOUNT_ID: str = _cfg["CHATWOOT_ACCOUNT_ID"]

PORT: int = int(os.getenv("PORT", "8000"))
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
