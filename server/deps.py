"""Dependencias FastAPI: servicios del backend, usuario, organización activa y cuotas."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from openai import AsyncOpenAI

from cima_core.cache import Cache, set_default_cache

from .auth import AuthError, AuthUser, TokenVerifier
from .cache import SupabaseCache
from .config import Settings, load_settings
from .store import SupabaseStore, minute_ago
from .supabase_rest import SupabaseRest


@dataclass
class Backend:
    settings: Settings
    store: Optional[SupabaseStore]
    verifier: Optional[TokenVerifier]
    cache: Optional[Cache]

    def openai_client(self) -> Any:
        if not self.settings.openai_api_key:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "OpenAI no está configurado")
        return AsyncOpenAI(api_key=self.settings.openai_api_key)

    def app_url(self, request: Request) -> str:
        return self.settings.app_url or str(request.base_url).rstrip("/")


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    settings = load_settings()
    if not settings.supabase_configured:
        return Backend(settings=settings, store=None, verifier=None, cache=None)
    rest = SupabaseRest(settings.supabase_url, settings.supabase_secret_key,
                        settings.supabase_publishable_key)
    cache = SupabaseCache(rest)
    # Componentes del núcleo que no reciben caché explícita (búsqueda de
    # formulación/prospecto) usan también la tabla compartida
    set_default_cache(cache)
    return Backend(settings=settings, store=SupabaseStore(rest), verifier=TokenVerifier(rest), cache=cache)


def require_store(backend: Backend = Depends(get_backend)) -> SupabaseStore:
    if backend.store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Supabase no está configurado")
    return backend.store


async def current_user(authorization: Optional[str] = Header(default=None),
                       backend: Backend = Depends(get_backend)) -> AuthUser:
    if backend.verifier is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Supabase no está configurado")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Inicie sesión para continuar",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return await backend.verifier.verify(authorization[7:].strip())
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e),
                            headers={"WWW-Authenticate": "Bearer"}) from e


@dataclass(frozen=True)
class OrgContext:
    user: AuthUser
    org_id: str
    role: str


def parse_uuid(value: str, what: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{what} no válido")


async def membership(org_id: str, user: AuthUser, store: SupabaseStore) -> OrgContext:
    role = await store.get_membership_role(org_id, user.id)
    if role is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No pertenece a esta organización")
    return OrgContext(user=user, org_id=org_id, role=role)


async def current_org(x_org_id: Optional[str] = Header(default=None),
                      user: AuthUser = Depends(current_user),
                      store: SupabaseStore = Depends(require_store)) -> OrgContext:
    """Organización activa (cabecera X-Org-Id), verificada contra las membresías."""
    if not x_org_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Falta la cabecera X-Org-Id")
    return await membership(parse_uuid(x_org_id, "X-Org-Id"), user, store)


async def enforce_quota(ctx: OrgContext, store: SupabaseStore, settings: Settings) -> None:
    """Cuota mensual de la organización y límite por minuto del usuario."""
    quota = await store.get_monthly_quota(ctx.org_id)
    if await store.org_requests_this_month(ctx.org_id) >= quota:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "La organización ha agotado su cuota mensual de consultas")
    if await store.user_requests_since(ctx.user.id, minute_ago()) >= settings.user_requests_per_minute:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            "Demasiadas peticiones seguidas; espere un minuto",
                            headers={"Retry-After": "60"})
