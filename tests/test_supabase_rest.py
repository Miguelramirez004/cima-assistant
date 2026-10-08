"""Forma de las peticiones HTTP a Supabase (PostgREST/Auth) y caché compartida."""

from datetime import datetime, timezone

import httpx
import pytest

from server.cache import SupabaseCache
from server.store import SupabaseStore
from server.supabase_rest import SupabaseError, SupabaseRest


class Recorder:
    def __init__(self, responses):
        self.requests = []
        self.responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def rest_with(*responses):
    recorder = Recorder(responses)
    rest = SupabaseRest("https://proj.supabase.co/", "sb_secret_x", "sb_publishable_y",
                        transport=httpx.MockTransport(recorder))
    return rest, recorder


async def test_secret_key_sent_as_apikey_only():
    rest, rec = rest_with(httpx.Response(200, json=[{"role": "admin"}]))
    role = await SupabaseStore(rest).get_membership_role("org-1", "user-1")
    request = rec.requests[0]
    assert role == "admin"
    assert request.headers["apikey"] == "sb_secret_x"
    assert "authorization" not in request.headers
    assert request.url.path == "/rest/v1/organization_members"
    assert request.url.params["organization_id"] == "eq.org-1"
    assert request.url.params["user_id"] == "eq.user-1"


async def test_count_reads_content_range_and_encodes_timestamps():
    rest, rec = rest_with(httpx.Response(200, json=[], headers={"content-range": "*/7"}))
    since = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    assert await SupabaseStore(rest).user_requests_since("user-1", since) == 7
    request = rec.requests[0]
    assert request.headers["prefer"] == "count=exact"
    # '+' must be percent-encoded or PostgREST would read it as a space
    assert "%2B00%3A00" in str(request.url) or "%2B00:00" in str(request.url)
    assert request.url.params["created_at"] == "gte.2026-10-08T12:00:00+00:00"


async def test_errors_raise_supabase_error():
    rest, _ = rest_with(httpx.Response(409, json={"message": "duplicate key value"}))
    with pytest.raises(SupabaseError) as exc:
        await SupabaseStore(rest).create_organization("A", "a", 10)
    assert exc.value.status == 409


async def test_auth_user_check_uses_user_token():
    rest, rec = rest_with(httpx.Response(200, json={"id": "u1"}))
    await rest.get_user("user-jwt")
    request = rec.requests[0]
    assert request.headers["authorization"] == "Bearer user-jwt"
    assert request.headers["apikey"] == "sb_publishable_y"


async def test_invite_passes_redirect():
    rest, rec = rest_with(httpx.Response(200, json={}))
    await rest.invite_user("a@b.test", "https://app.test/invite/tok")
    request = rec.requests[0]
    assert request.url.path == "/auth/v1/invite"
    assert request.url.params["redirect_to"] == "https://app.test/invite/tok"


async def test_cache_roundtrip_and_upsert():
    rest, rec = rest_with(httpx.Response(201), httpx.Response(200, json=[{"value": {"a": 1}}]))
    cache = SupabaseCache(rest)
    await cache.set("maestras:1:x", {"a": 1}, ttl=60)
    assert await cache.get("maestras:1:x") == {"a": 1}
    upsert, read = rec.requests
    assert "resolution=merge-duplicates" in upsert.headers["prefer"]
    assert read.url.params["expires_at"].startswith("gt.")


async def test_cache_failures_are_swallowed():
    rest, _ = rest_with(httpx.Response(500, json={"message": "down"}), httpx.Response(500, json={}))
    cache = SupabaseCache(rest)
    assert await cache.get("k") is None
    await cache.set("k", 1)
