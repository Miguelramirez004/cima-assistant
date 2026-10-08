"""Núcleo de CIMA Assistant: recuperación sobre la API CIMA (AEMPS) y redacción con OpenAI."""

from .cache import Cache, MemoryCache, NullCache, get_default_cache, set_default_cache
from .models import ChatTurn, ConsultaResult, FormulacionResult, ProspectoResult, Reference
from .service import run_consulta, run_formulacion, run_prospecto

__all__ = [
    "Cache", "MemoryCache", "NullCache", "get_default_cache", "set_default_cache",
    "ChatTurn", "ConsultaResult", "FormulacionResult", "ProspectoResult", "Reference",
    "run_consulta", "run_formulacion", "run_prospecto",
]
