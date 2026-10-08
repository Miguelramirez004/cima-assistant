"""
Punto de entrada de la API en Vercel (Python Function).

Vercel enruta /api/* a este módulo (ver next.config.ts y vercel.json); en
desarrollo se sirve con `uvicorn api.index:app` en :8000. La aplicación vive
en el paquete `server` (fuera de api/, porque Vercel convierte cada archivo
de api/ en una función independiente).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.app import app  # noqa: E402,F401
