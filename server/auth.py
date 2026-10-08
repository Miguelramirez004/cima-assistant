"""
Verificación de los tokens de acceso de Supabase Auth.

- Claves de firma asimétricas (ES256/RS256, las de los proyectos nuevos): se
  verifican localmente con las claves públicas del JWKS del proyecto, en
  caché unos minutos. Sin viaje de red por petición.
- Secreto compartido (HS256, proyectos antiguos): se valida contra el
  servidor de Auth (`GET /auth/v1/user`), como recomienda Supabase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jwt

from .supabase_rest import SupabaseError, SupabaseRest

JWKS_TTL_SECONDS = 600
ASYMMETRIC_ALGORITHMS = ("ES256", "RS256")


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: Optional[str]


class TokenVerifier:
    def __init__(self, rest: SupabaseRest):
        self.rest = rest
        self.issuer = f"{rest.url}/auth/v1"
        self._jwks: Dict[str, Dict[str, Any]] = {}
        self._jwks_fetched_at = 0.0

    async def verify(self, token: str) -> AuthUser:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as e:
            raise AuthError("Token mal formado") from e

        alg = header.get("alg")
        kid = header.get("kid")
        if alg in ASYMMETRIC_ALGORITHMS and kid:
            return await self._verify_locally(token, alg, kid)
        return await self._verify_with_auth_server(token)

    async def _verify_locally(self, token: str, alg: str, kid: str) -> AuthUser:
        jwk = await self._get_jwk(kid)
        if jwk is None:
            raise AuthError("Clave de firma desconocida")
        try:
            claims = jwt.decode(
                token, jwt.PyJWK(jwk).key, algorithms=[alg],
                audience="authenticated", issuer=self.issuer,
                options={"require": ["exp", "sub"]},
            )
        except jwt.PyJWTError as e:
            raise AuthError("Token no válido o caducado") from e
        if claims.get("role") != "authenticated":
            raise AuthError("Se requiere un usuario autenticado")
        return AuthUser(id=claims["sub"], email=claims.get("email"))

    async def _get_jwk(self, kid: str) -> Optional[Dict[str, Any]]:
        age = time.monotonic() - self._jwks_fetched_at
        # Refrescar al caducar, o ante un kid desconocido (rotación de claves)
        # como mucho cada 30 s, para que tokens con kids inventados no
        # provoquen una petición al JWKS cada vez
        if age > JWKS_TTL_SECONDS or (kid not in self._jwks and age > 30):
            try:
                data = await self.rest.get_jwks()
            except SupabaseError as e:
                raise AuthError("No se pudieron obtener las claves de firma") from e
            self._jwks = {k["kid"]: k for k in data.get("keys", []) if k.get("kid")}
            self._jwks_fetched_at = time.monotonic()
        return self._jwks.get(kid)

    async def _verify_with_auth_server(self, token: str) -> AuthUser:
        try:
            user = await self.rest.get_user(token)
        except SupabaseError as e:
            raise AuthError("Token no válido o caducado") from e
        if not user.get("id"):
            raise AuthError("Token no válido")
        return AuthUser(id=user["id"], email=user.get("email"))
