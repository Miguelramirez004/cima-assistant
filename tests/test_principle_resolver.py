from cima_core.cache import MemoryCache
from cima_core.principle_resolver import ActivePrincipleResolver


async def test_resolution_is_cached_across_instances(monkeypatch):
    cache = MemoryCache()
    calls = []

    async def fake_fetch(self, session, term):
        calls.append(term)
        return [{"id": 42, "nombre": "IBUPROFENO"}]

    monkeypatch.setattr(ActivePrincipleResolver, "_fetch_master_items", fake_fetch)

    first = await ActivePrincipleResolver(cache=cache).resolve(None, "ibuprofeno")
    # Otra instancia (otra petición serverless) reutiliza la caché compartida
    second = await ActivePrincipleResolver(cache=cache).resolve(None, "Ibuprofeno")

    assert first.id == second.id == 42
    assert second.nombre == "IBUPROFENO"
    assert calls == ["ibuprofeno"]


async def test_failed_resolution_is_cached(monkeypatch):
    cache = MemoryCache()
    calls = []

    async def fake_fetch(self, session, term):
        calls.append(term)
        return []

    monkeypatch.setattr(ActivePrincipleResolver, "_fetch_master_items", fake_fetch)
    resolver = ActivePrincipleResolver(cache=cache)
    assert await resolver.resolve(None, "noexisteprincipio") is None
    assert await resolver.resolve(None, "noexisteprincipio") is None
    assert len(calls) == 1
