"""Configuración del backend, leída de variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


@dataclass(frozen=True)
class Settings:
    supabase_url: Optional[str]
    supabase_publishable_key: Optional[str]
    supabase_secret_key: Optional[str]
    openai_api_key: Optional[str]
    # URL pública de la app (enlaces de invitación). Si falta, se usa el
    # origen de la petición.
    app_url: Optional[str]
    # Peticiones con OpenAI permitidas por usuario y minuto
    user_requests_per_minute: int

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key)


def load_settings() -> Settings:
    url = _env("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL")
    return Settings(
        supabase_url=url.rstrip("/") if url else None,
        supabase_publishable_key=_env(
            "SUPABASE_PUBLISHABLE_KEY", "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
            "NEXT_PUBLIC_SUPABASE_ANON_KEY",
        ),
        supabase_secret_key=_env("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"),
        openai_api_key=_env("OPENAI_API_KEY"),
        app_url=(_env("APP_URL") or "").rstrip("/") or None,
        user_requests_per_minute=int(_env("USER_REQUESTS_PER_MINUTE", default="10")),
    )
