"""Caché de la API CIMA en la tabla `cima_cache`, compartida por todas las instancias."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from cima_core.cache import DEFAULT_TTL_SECONDS

from .supabase_rest import SupabaseRest

logger = logging.getLogger(__name__)


class SupabaseCache:
    """Implementa cima_core.cache.Cache. Un fallo de la caché nunca rompe la petición."""

    def __init__(self, rest: SupabaseRest):
        self.rest = rest

    async def get(self, key: str) -> Optional[Any]:
        try:
            rows = await self.rest.select("cima_cache", {
                "select": "value",
                "key": f"eq.{key}",
                "expires_at": f"gt.{datetime.now(timezone.utc).isoformat()}",
            })
        except Exception as e:
            logger.warning(f"cima_cache get failed for {key}: {e}")
            return None
        return rows[0]["value"] if rows else None

    async def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        try:
            await self.rest.insert("cima_cache", {
                "key": key, "value": value, "expires_at": expires_at.isoformat(),
            }, upsert=True, returning=False)
        except Exception as e:
            logger.warning(f"cima_cache set failed for {key}: {e}")
