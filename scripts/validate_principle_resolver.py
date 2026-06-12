"""
Validación del resolutor de principios activos contra la API REAL de CIMA.

Este entorno de desarrollo no tiene acceso de red a cima.aemps.es, así que la
integración se probó con la API simulada. Ejecuta este script en tu máquina
(con red) para validar el comportamiento end-to-end antes de desplegar:

    python scripts/validate_principle_resolver.py

Comprueba:
1. Resolución término -> idpractiv1 vía maestras?maestra=1 (incluye fármacos
   con 'A' que el antiguo ranking penalizaba, acentos y un término inexistente).
2. Recuperación exacta con medicamentos?idpractiv1=... y que TODOS los
   resultados contienen el principio resuelto (cero "relleno alfabético").
3. El flujo completo MedicationSearchGraph.execute_search con una consulta real.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from principle_resolver import ActivePrincipleResolver  # noqa: E402
from search_graph import MedicationSearchGraph  # noqa: E402

TERMS = [
    "ibuprofeno",
    "amoxicilina",            # 'A': antes autopenalizada por el parche abacavir
    "atorvastatina",          # 'A'
    "acido acetilsalicilico", # acentos / multi-palabra
    "metamizol",
    "noexisteprincipio",      # debe devolver None, sin excepciones
]


async def main() -> int:
    failures = 0
    resolver = ActivePrincipleResolver()

    async with aiohttp.ClientSession() as session:
        print("== 1. Resolución término -> idpractiv1 (maestras?maestra=1) ==")
        resolved_ids = {}
        for term in TERMS:
            result = await resolver.resolve(session, term)
            if term == "noexisteprincipio":
                ok = result is None
            else:
                ok = result is not None
            status = "OK " if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"  [{status}] {term!r} -> {result}")
            if result:
                resolved_ids[term] = result

        print("\n== 2. Recuperación exacta medicamentos?idpractiv1=... ==")
        probe = resolved_ids.get("amoxicilina") or next(iter(resolved_ids.values()), None)
        if probe is None:
            print("  [FAIL] No se resolvió ningún principio; no se puede probar la búsqueda")
            failures += 1
        else:
            url = "https://cima.aemps.es/cima/rest/medicamentos"
            params = {"idpractiv1": str(probe.id), "pagina": "1"}
            async with session.get(url, params=params, headers={"Accept": "application/json"}) as resp:
                data = await resp.json()
            meds = data.get("resultados", []) if isinstance(data, dict) else []
            print(f"  idpractiv1={probe.id} ({probe.nombre}): {len(meds)} resultados")
            principle = probe.nombre.lower().split()[0]
            bad = [m for m in meds if principle not in (m.get("pactivos") or "").lower()]
            if meds and not bad:
                print(f"  [OK ] Todos los resultados contienen '{principle}' en pactivos")
            else:
                failures += 1
                print(f"  [FAIL] {len(bad)} resultados sin el principio (o lista vacía)")
                for m in bad[:3]:
                    print(f"         - {m.get('nombre')} | pactivos={m.get('pactivos')}")

    print("\n== 3. Flujo completo execute_search ==")
    graph = MedicationSearchGraph()
    results, quality, _intent = await graph.execute_search(
        "Suspensión de amoxicilina 250mg/5ml para uso pediátrico"
    )
    print(f"  quality={quality}, resultados={len(results)}")
    for r in results[:5]:
        print(f"   - {r.get('nregistro')} {r.get('nombre')} (score={r.get('relevance_score')})")
    if quality in ("very_high", "high") and results:
        print("  [OK ] Recuperación por id con calidad alta")
    else:
        failures += 1
        print("  [FAIL] La búsqueda no usó la ruta por id o no devolvió resultados")

    print(f"\n{'TODO OK' if failures == 0 else f'{failures} FALLOS'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
