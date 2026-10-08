from cima_core.cache import MemoryCache, NullCache, get_default_cache, set_default_cache


async def test_memory_cache_roundtrip_and_expiry(monkeypatch):
    cache = MemoryCache()
    await cache.set("k", {"a": 1}, ttl=10)
    assert await cache.get("k") == {"a": 1}

    import cima_core.cache as cache_module
    real = cache_module.time.monotonic
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: real() + 11)
    assert await cache.get("k") is None


async def test_memory_cache_evicts_when_full():
    cache = MemoryCache(max_entries=2)
    await cache.set("a", 1, ttl=1)
    await cache.set("b", 2, ttl=100)
    await cache.set("c", 3, ttl=100)
    assert await cache.get("a") is None
    assert await cache.get("b") == 2 and await cache.get("c") == 3


async def test_null_cache_and_default_override():
    original = get_default_cache()
    try:
        set_default_cache(NullCache())
        await get_default_cache().set("k", 1)
        assert await get_default_cache().get("k") is None
    finally:
        set_default_cache(original)
