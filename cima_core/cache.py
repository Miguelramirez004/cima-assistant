"""
Caché de respuestas de la API CIMA, desacoplada del proceso.

En Streamlit bastaba con diccionarios en memoria porque el servidor vivía
mucho tiempo. En Vercel cada función es efímera, así que la caché se inyecta:

- MemoryCache: TTL en memoria del proceso (tests, desarrollo, legacy).
- SupabaseCache (fase 4): tabla `cima_cache` compartida por todas las
  instancias.

Los valores deben ser serializables a JSON. Los datos de CIMA son públicos
(no son datos de usuario ni de organización), por eso la caché es global.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol, Tuple

# TTL por defecto: los catálogos y fichas de la AEMPS cambian con poca frecuencia
DEFAULT_TTL_SECONDS = 24 * 3600


class Cache(Protocol):
    async def get(self, key: str) -> Optional[Any]:
        """Devuelve el valor almacenado o None si no existe o ha caducado."""
        ...

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        ...


class MemoryCache:
    """Caché TTL en memoria, acotada en número de entradas."""

    def __init__(self, max_entries: int = 2048):
        self.max_entries = max_entries
        self._data: Dict[str, Tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        if len(self._data) >= self.max_entries and key not in self._data:
            # Expulsar la entrada que caduca antes
            oldest = min(self._data, key=lambda k: self._data[k][0])
            self._data.pop(oldest, None)
        self._data[key] = (time.monotonic() + ttl, value)


class NullCache:
    """No almacena nada (útil para forzar llamadas reales en validaciones)."""

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        return None


_default_cache: Cache = MemoryCache()


def get_default_cache() -> Cache:
    """Caché usada por los componentes que no reciben una explícitamente."""
    return _default_cache


def set_default_cache(cache: Cache) -> None:
    """Sustituye la caché por defecto (p. ej. por SupabaseCache al arrancar la API)."""
    global _default_cache
    _default_cache = cache
