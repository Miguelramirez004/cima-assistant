"""
Utilidades de seguridad para CIMA Assistant.

Centraliza el endurecimiento que se aplica de forma consistente en los flujos de
formulación, prospectos y chat:

- Saneamiento de salida para sinks de Streamlit con ``unsafe_allow_html=True``
  (mitigación de XSS almacenado/reflejado).
- Neutralización de inyección de prompts en el contenido recuperado de CIMA antes
  de incrustarlo en el contexto del modelo.
- Limpieza de HTML en tiempo lineal con límites de tamaño (mitigación de ReDoS y
  coste descontrolado).
- Acotado de la entrada de usuario (límite de longitud).

Todo el contenido procedente del modelo, de la API de CIMA o de páginas web
externas debe tratarse como NO confiable y pasar por estas funciones antes de
llegar a un sink HTML o al contexto del LLM.
"""

from __future__ import annotations

import html
import re

# Límites duros para acotar el trabajo sobre texto no confiable / generado por el
# modelo (DoS / ReDoS / coste de tokens).
MAX_QUERY_CHARS = 2000
MAX_HTML_CHARS = 200_000
MAX_CONTEXT_CHARS = 60_000

# Una única pasada lineal: <[^>]*> no tiene cuantificadores anidados, por lo que
# no es susceptible de backtracking catastrófico.
_ANY_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")

# Frases habitualmente utilizadas para secuestrar un LLM mediante contenido
# recuperado o envenenado (inyección indirecta de prompts). Se sustituyen por un
# marcador inerte; no se intenta "entender", solo desactivar el patrón.
_INJECTION_MARKERS = re.compile(
    r"(?i)("
    r"ignora(?:r|s|)\s+(?:todo|todas?\s+las\s+instrucciones|lo\s+anterior|las\s+reglas)|"
    r"ignore\s+(?:all|previous|the\s+above|prior)\s+(?:instructions?|prompts?)?|"
    r"olvida(?:r|)\s+(?:todo|las\s+instrucciones|lo\s+anterior)|"
    r"forget\s+(?:all|everything|previous|the\s+above)|"
    r"disregard\s+(?:all|previous|the\s+above)|"
    r"system\s*prompt|prompt\s+del\s+sistema|"
    r"nuevas\s+instrucciones|new\s+instructions|"
    r"act[uú]a\s+como\s+si|act\s+as\s+(?:a|an|if)|"
    r"</?\s*(?:system|assistant|user|im_start|im_end)\s*>?"
    r")"
)


def clamp_query(text: object, limit: int = MAX_QUERY_CHARS) -> str:
    """Acota la longitud de una consulta de usuario para limitar coste y abuso."""
    if not text:
        return ""
    return str(text)[:limit]


def escape_html(text: object) -> str:
    """
    Escapa texto para insertarlo de forma segura en un sink ``unsafe_allow_html``.

    Preserva los saltos de línea (los convierte en ``<br>``) para que un prospecto
    de texto plano siga leyéndose correctamente, pero neutraliza cualquier etiqueta
    HTML/JS que pudiera haber generado el modelo o haberse colado desde la consulta.
    """
    if not text:
        return ""
    escaped = html.escape(str(text), quote=True)
    return escaped.replace("\n", "<br>")


def strip_html(text: object, max_chars: int = MAX_HTML_CHARS) -> str:
    """
    Elimina HTML de un documento recuperado en tiempo lineal.

    Acota primero el tamaño de entrada para evitar trabajo excesivo sobre páginas
    adversarias y devuelve texto plano colapsando espacios en blanco.
    """
    if not text:
        return ""
    text = str(text)[:max_chars]
    text = _ANY_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def neutralize_injection(text: object, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Desactiva intentos de inyección de prompts en contenido no confiable antes de
    colocarlo en el contexto del modelo. Acota tamaño y sustituye marcadores de
    instrucción por un texto inerte; no bloquea, solo los hace inocuos.
    """
    if not text:
        return ""
    text = str(text)[:max_chars]
    return _INJECTION_MARKERS.sub("[contenido omitido]", text)


def clean_retrieved_text(text: object) -> str:
    """Atajo: limpia HTML y neutraliza inyección sobre contenido recuperado de CIMA."""
    return neutralize_injection(strip_html(text))


def safe_url(url: object) -> str:
    """
    Devuelve una URL segura para usar en un atributo ``href`` dentro de un sink
    ``unsafe_allow_html``. Solo se permiten esquemas http/https; cualquier otra
    cosa (javascript:, data:, etc.) se descarta. El resultado va escapado.
    """
    if not url:
        return "#"
    url = str(url).strip()
    if not re.match(r"(?i)^https?://", url):
        return "#"
    return html.escape(url, quote=True)
