"""
Utilidades comunes para trabajar con las respuestas de la API REST de CIMA
(AEMPS), según la especificación v1.23:

- Los documentos asociados a un medicamento llegan en `docs[]` con
  `tipo` (1: ficha técnica, 2: prospecto), `url` (PDF) y `urlHtml`
  (solo si `secc=true`). Esas URLs oficiales deben preferirse a construirlas
  a mano.
- Las fechas se devuelven como Unix epoch (POSIX time, GMT+2), no como
  cadenas "yyyymmdd".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import tiktoken

logger = logging.getLogger(__name__)

# Huso indicado en la especificación de la API para las fechas epoch
CIMA_TZ = timezone(timedelta(hours=2))

DOC_TYPE_FICHA_TECNICA = 1
DOC_TYPE_PROSPECTO = 2


def extract_doc_urls(med: Optional[Dict[str, Any]], nregistro: str) -> Dict[str, str]:
    """
    Obtiene las URLs de ficha técnica y prospecto de un medicamento.

    Prefiere las URLs oficiales que la propia API devuelve en `docs[]`
    (`urlHtml` si el documento está segmentado, `url` en su defecto) y solo
    como último recurso construye las rutas HTML documentadas
    (`.../ft/{nregistro}/FichaTecnica.html`, `.../p/{nregistro}/Prospecto.html`).
    """
    urls = {
        "ficha_tecnica": f"https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FichaTecnica.html",
        "prospecto": f"https://cima.aemps.es/cima/dochtml/p/{nregistro}/Prospecto.html",
    }
    docs = (med or {}).get("docs")
    if not isinstance(docs, list):
        return urls

    type_to_key = {DOC_TYPE_FICHA_TECNICA: "ficha_tecnica", DOC_TYPE_PROSPECTO: "prospecto"}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        key = type_to_key.get(doc.get("tipo"))
        if not key:
            continue
        url = doc.get("urlHtml") or doc.get("url")
        if isinstance(url, str) and url.startswith("http"):
            urls[key] = url
    return urls


def format_cima_date(value: Any) -> str:
    """
    Formatea una fecha devuelta por la API CIMA como "dd/mm/yyyy".

    La especificación define las fechas como Unix epoch GMT+2; en la práctica la
    API las emite en milisegundos. Se aceptan epoch en segundos o milisegundos
    (int, float o cadena numérica). Si el valor no es interpretable se devuelve
    tal cual en forma de cadena, sin inventar nada.
    """
    if value is None or value == "":
        return ""

    epoch: Optional[float] = None
    if isinstance(value, (int, float)):
        epoch = float(value)
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        epoch = float(value.strip())

    if epoch is not None:
        try:
            if abs(epoch) >= 1e12:  # milisegundos
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, tz=CIMA_TZ).strftime("%d/%m/%Y")
        except (OverflowError, OSError, ValueError) as e:
            logger.warning(f"Unparseable CIMA epoch date {value!r}: {e}")

    return str(value)

def get_tokenizer(model: str):
    """
    Tokenizador para contar tokens con degradación robusta.

    `tiktoken.encoding_for_model` descarga el fichero BPE en el primer uso, así
    que puede fallar con errores de red (no solo KeyError) en despliegues con
    salida restringida. Se intenta el encoding del modelo, luego cl100k_base y,
    si nada está disponible, se devuelve None (el contador aproximará).
    """
    for getter in (lambda: tiktoken.encoding_for_model(model),
                   lambda: tiktoken.get_encoding("cl100k_base")):
        try:
            return getter()
        except Exception as e:
            logger.warning(f"Tokenizer unavailable ({e.__class__.__name__}); trying fallback")
    return None


def count_tokens(tokenizer, text: str) -> int:
    """Cuenta tokens con el tokenizador dado, o aproxima (~4 chars/token) sin él."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)
