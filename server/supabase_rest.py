"""
Cliente HTTP mínimo para Supabase (PostgREST y Auth) con la clave secreta.

Con las claves nuevas (`sb_secret_...`) basta la cabecera `apikey`: la pasarela
de Supabase sintetiza el `Authorization` de service_role. Las claves legacy
(JWT service_role) funcionan igual. La clave secreta salta RLS: este cliente
solo se usa en el servidor y siempre con la organización ya verificada.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx


class SupabaseError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"Supabase {status}: {message}")
        self.status = status
        self.message = message


class SupabaseRest:
    def __init__(self, url: str, secret_key: str, publishable_key: Optional[str] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.secret_key = secret_key
        self.publishable_key = publishable_key
        self._transport = transport
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _http(self) -> httpx.AsyncClient:
        # Un cliente por event loop (Vercel reutiliza la instancia entre peticiones)
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.url, transport=self._transport, timeout=self._timeout,
                headers={"apikey": self.secret_key},
            )
            self._loop = loop
        return self._client

    @staticmethod
    def _check(response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            try:
                body = response.json()
                message = body.get("message") or body.get("msg") or body.get("error_description") or str(body)
            except Exception:
                message = response.text[:300]
            raise SupabaseError(response.status_code, message)
        return response

    # ------------------------------------------------------------- PostgREST

    async def select(self, table: str, params: Dict[str, str]) -> List[Dict[str, Any]]:
        response = await self._http().get(f"/rest/v1/{table}", params=params)
        return self._check(response).json()

    async def count(self, table: str, params: Dict[str, str]) -> int:
        response = await self._http().get(
            f"/rest/v1/{table}", params={**params, "select": "*", "limit": "0"},
            headers={"Prefer": "count=exact"},
        )
        content_range = self._check(response).headers.get("content-range", "*/0")
        return int(content_range.rsplit("/", 1)[-1])

    async def insert(self, table: str, rows: Any, *, upsert: bool = False,
                     returning: bool = True) -> List[Dict[str, Any]]:
        prefer = ["return=representation" if returning else "return=minimal"]
        if upsert:
            prefer.append("resolution=merge-duplicates")
        response = await self._http().post(
            f"/rest/v1/{table}", json=rows, headers={"Prefer": ",".join(prefer)},
        )
        self._check(response)
        return response.json() if returning else []

    async def rpc(self, function: str, args: Dict[str, Any]) -> Any:
        response = await self._http().post(f"/rest/v1/rpc/{function}", json=args)
        return self._check(response).json()

    # ------------------------------------------------------------------ Auth

    async def get_user(self, access_token: str) -> Dict[str, Any]:
        """Valida un token de usuario contra Auth (JWT con secreto compartido)."""
        response = await self._http().get(
            "/auth/v1/user",
            headers={"apikey": self.publishable_key or self.secret_key,
                     "Authorization": f"Bearer {access_token}"},
        )
        return self._check(response).json()

    async def get_jwks(self) -> Dict[str, Any]:
        response = await self._http().get("/auth/v1/.well-known/jwks.json")
        return self._check(response).json()

    async def invite_user(self, email: str, redirect_to: str) -> None:
        """Email de invitación de Supabase Auth (crea el usuario)."""
        response = await self._http().post(
            "/auth/v1/invite", params={"redirect_to": redirect_to}, json={"email": email},
        )
        self._check(response)

    async def send_magic_link(self, email: str, redirect_to: str) -> None:
        """Enlace de acceso para un usuario ya existente (no crea usuarios)."""
        response = await self._http().post(
            "/auth/v1/otp", params={"redirect_to": redirect_to},
            json={"email": email, "create_user": False},
        )
        self._check(response)
