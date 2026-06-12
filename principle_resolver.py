"""
Resolución de principios activos contra el catálogo oficial de CIMA.

Implementa el "punto 1" de la mejora del RAG: en lugar de adivinar el principio
activo con listas hardcodeadas y búsquedas difusas por nombre (origen del
"problema abacavir"), se resuelve el término del usuario contra la maestra
oficial de principios activos de la AEMPS:

    GET {base}/maestras?maestra=1&nombre={termino}

(CIMA REST API v1.23, sección "GET maestras?{condiciones}"; maestra=1 es el
catálogo de principios activos y cada ítem devuelve {id, codigo, nombre}).

El `id` resultante se usa después en:

    GET {base}/medicamentos?idpractiv1={id}

que recupera por identificador exacto, eliminando los falsos positivos
alfabéticos en origen.
"""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

from config import Config

logger = logging.getLogger(__name__)

# Máximo de candidatos de una consulta que se intentan resolver contra la
# maestra (limita latencia y carga sobre la API).
MAX_CANDIDATES = 3


@dataclass(frozen=True)
class ResolvedPrinciple:
    """Principio activo resuelto contra la maestra oficial de CIMA."""
    id: int
    nombre: str          # Nombre oficial en el catálogo (p. ej. "IBUPROFENO")
    matched_term: str    # Término de la consulta que produjo la resolución


def _normalize(text: str) -> str:
    """Minúsculas sin acentos, para comparar términos con el catálogo oficial."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


class ActivePrincipleResolver:
    """
    Resuelve términos de usuario a IDs oficiales de principio activo.

    Mantiene una caché en memoria por proceso (los catálogos de la AEMPS cambian
    con muy poca frecuencia). La resolución fallida también se cachea para no
    repetir llamadas inútiles dentro de la misma sesión.
    """

    def __init__(self, base_url: str = Config.CIMA_BASE_URL):
        self.base_url = base_url
        self._cache: Dict[str, Optional[ResolvedPrinciple]] = {}

    async def resolve(self, session: aiohttp.ClientSession, term: str) -> Optional[ResolvedPrinciple]:
        """
        Resuelve un único término contra la maestra de principios activos.
        Devuelve None si no hay coincidencia razonable.
        """
        key = _normalize(term)
        if not key or len(key) < 3:
            return None
        if key in self._cache:
            return self._cache[key]

        items = await self._fetch_master_items(session, term)
        resolved = self._pick_best(key, term, items)
        self._cache[key] = resolved
        if resolved:
            logger.info(f"Resolved active principle '{term}' -> id={resolved.id} ({resolved.nombre})")
        return resolved

    async def resolve_candidates(self, session: aiohttp.ClientSession,
                                 candidates: List[str]) -> Optional[ResolvedPrinciple]:
        """
        Intenta resolver una lista de candidatos (en orden de prioridad) y
        devuelve la primera resolución con éxito. Las llamadas se lanzan en
        paralelo pero se respeta la prioridad del orden de entrada.
        """
        # Deduplicar conservando el orden y acotar el número de llamadas.
        seen = set()
        unique: List[str] = []
        for cand in candidates:
            norm = _normalize(cand)
            if norm and len(norm) >= 3 and norm not in seen:
                seen.add(norm)
                unique.append(cand)
            if len(unique) >= MAX_CANDIDATES:
                break

        if not unique:
            return None

        results = await asyncio.gather(
            *(self.resolve(session, cand) for cand in unique),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, ResolvedPrinciple):
                return result
        return None

    async def _fetch_master_items(self, session: aiohttp.ClientSession, term: str) -> List[Dict[str, Any]]:
        """Llama a maestras?maestra=1&nombre=... y devuelve la lista de ítems."""
        url = f"{self.base_url}/maestras"
        params = {"maestra": "1", "nombre": term, "pagina": "1"}
        headers = {"Accept": "application/json"}
        try:
            async with session.get(url, params=params, headers=headers, timeout=15) as response:
                if response.status != 200:
                    logger.warning(f"maestras lookup for '{term}' returned status {response.status}")
                    return []
                data = await response.json()
        except Exception as e:
            logger.warning(f"maestras lookup failed for '{term}': {str(e)}")
            return []

        # La API pagina las respuestas; ser tolerante con ambos envoltorios.
        if isinstance(data, dict):
            items = data.get("resultados", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        return [item for item in items if isinstance(item, dict) and item.get("id") and item.get("nombre")]

    def _pick_best(self, normalized_term: str, original_term: str,
                   items: List[Dict[str, Any]]) -> Optional[ResolvedPrinciple]:
        """
        Elige el ítem del catálogo que mejor corresponde al término.

        Preferencia: igualdad exacta > el nombre empieza por el término > el
        término aparece como palabra dentro del nombre. En caso de empate gana
        el nombre más corto (el principio "simple" frente a sales/combinaciones,
        p. ej. IBUPROFENO antes que IBUPROFENO ARGININA).
        """
        exact: List[Dict[str, Any]] = []
        prefix: List[Dict[str, Any]] = []
        contains: List[Dict[str, Any]] = []

        for item in items:
            name_norm = _normalize(str(item["nombre"]))
            if name_norm == normalized_term:
                exact.append(item)
            elif name_norm.startswith(normalized_term):
                prefix.append(item)
            elif normalized_term in name_norm.split() or normalized_term in name_norm:
                contains.append(item)

        for bucket in (exact, prefix, contains):
            if bucket:
                best = min(bucket, key=lambda it: len(str(it["nombre"])))
                try:
                    return ResolvedPrinciple(
                        id=int(best["id"]),
                        nombre=str(best["nombre"]),
                        matched_term=original_term,
                    )
                except (TypeError, ValueError):
                    continue
        return None
