"""
API de CIMA Assistant (Vercel Python Function).

Vercel enruta /api/* a este módulo (ver vercel.json). En desarrollo se sirve
con uvicorn en :8000 y Next.js reenvía /api/* hacia él (ver next.config.ts).

Fase 1: solo el esqueleto y /api/health. Los endpoints de formulación,
consulta y prospecto (con autenticación Supabase y organización activa)
llegan en la fase 4 sobre cima_core.service.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI

# Permite importar cima_core tanto en Vercel como con `uvicorn api.index:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cima_core.config import Config  # noqa: E402

app = FastAPI(
    title="CIMA Assistant API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": Config.CHAT_MODEL,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }
