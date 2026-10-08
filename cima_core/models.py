"""
Resultados tipados de los tres casos de uso. Son el contrato entre el núcleo
Python y cualquier interfaz (API FastAPI en Vercel, app Streamlit legacy):
FastAPI los publica en su esquema OpenAPI y el frontend genera tipos a partir
de él.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Reference(BaseModel):
    """Fuente oficial citada en una respuesta."""
    title: str
    url: str
    nregistro: Optional[str] = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConsultaResult(BaseModel):
    answer: str
    reasoning: str = ""                 # traza del grafo de recuperación
    references: List[Reference] = Field(default_factory=list)
    success: bool = True


class FormulacionResult(BaseModel):
    answer: str
    context: str = ""                   # contexto CIMA usado para redactar
    references: List[Reference] = Field(default_factory=list)
    # "prospecto" cuando la consulta pide un prospecto y debe ir a ese flujo
    redirect: Optional[Literal["prospecto"]] = None
    success: bool = True


class ProspectoResult(BaseModel):
    content: str
    context: str = ""
    medication_name: Optional[str] = None
    nregistro: Optional[str] = None
    success: bool = True
