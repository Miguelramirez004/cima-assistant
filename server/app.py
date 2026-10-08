"""Aplicación FastAPI: /api/health, generación (formulación, consulta, prospecto) y organizaciones."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from cima_core.config import Config

from .deps import Backend, get_backend
from .routes import router
from .supabase_rest import SupabaseError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CIMA Assistant API",
    version="0.2.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.include_router(router)


@app.exception_handler(SupabaseError)
async def supabase_error_handler(request: Request, exc: SupabaseError) -> JSONResponse:
    logger.error(f"Supabase error on {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"detail": "Servicio de datos no disponible; inténtelo de nuevo"})


@app.get("/api/health")
async def health(backend: Backend = Depends(get_backend)) -> dict:
    settings = backend.settings
    return {
        "status": "ok",
        "model": Config.CHAT_MODEL,
        "openai_configured": bool(settings.openai_api_key),
        "supabase_configured": settings.supabase_configured,
    }
