"""
Puntos de entrada sin estado de los tres casos de uso.

Cada llamada crea su propio agente, lo ejecuta y cierra sus conexiones HTTP,
de modo que nada sobrevive entre peticiones salvo la caché (inyectable). Es
lo que necesitan las funciones serverless de Vercel y lo que usará la API.
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from .cache import Cache
from .cima_rag import CIMARagAgent
from .formulacion import FormulationAgent
from .models import ChatTurn, ConsultaResult, FormulacionResult, ProspectoResult, Reference
from .prospecto import ProspectoGenerator

logger = logging.getLogger(__name__)

FICHA_URL = "https://cima.aemps.es/cima/dochtml/ft/{nregistro}/FichaTecnica.html"

# [Ref 1: NOMBRE (Nº Registro: 12345)] — con o sin enlace markdown posterior
_REF_PATTERN = re.compile(r"\[Ref \d+: ([^()\]]+?) \(Nº Registro: (\d+)\)\]")
_NREGISTRO_PATTERN = re.compile(r"N(?:º|úmero de) [Rr]egistro:?\s*(\d+)")

PROSPECTO_REDIRECT_MESSAGE = (
    "Esta consulta solicita un prospecto. Utilice la sección «Prospectos» para "
    "generarlo según el formato oficial de la AEMPS."
)


def extract_references(text: str) -> List[Reference]:
    """Referencias únicas citadas en el texto con el formato [Ref X: Nombre (Nº Registro: N)]."""
    refs: List[Reference] = []
    seen = set()
    for name, nregistro in _REF_PATTERN.findall(text or ""):
        if nregistro in seen:
            continue
        seen.add(nregistro)
        refs.append(Reference(title=name.strip(), url=FICHA_URL.format(nregistro=nregistro),
                              nregistro=nregistro))
    return refs


def _history_dicts(history: Optional[Sequence[Any]]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, ChatTurn):
            turns.append(turn.model_dump())
        elif isinstance(turn, dict):
            turns.append({"role": turn.get("role"), "content": turn.get("content")})
    return turns


def _to_reference(ref: Dict[str, Any]) -> Reference:
    match = re.search(r"/ft/(\w+)/", ref.get("url", ""))
    return Reference(title=ref.get("title", ""), url=ref.get("url", ""),
                     nregistro=match.group(1) if match else None)


async def stream_consulta(question: str, history: Optional[Sequence[Any]] = None, *,
                          openai_client: AsyncOpenAI,
                          cache: Optional[Cache] = None) -> AsyncIterator[Dict[str, Any]]:
    """
    Consulta CIMA en streaming. Emite los eventos de `CIMARagAgent.ask_stream`
    con las referencias ya normalizadas ({title, url, nregistro}); el último
    evento es {"type": "done", ...} con la respuesta completa y el `usage`.
    """
    agent = CIMARagAgent(openai_client, cache=cache)
    try:
        async for event in agent.ask_stream(question, history=_history_dicts(history)):
            if "references" in event:
                event = {**event, "references": [_to_reference(r).model_dump() for r in event["references"]]}
            yield event
    finally:
        await agent.close()


async def run_consulta(question: str, history: Optional[Sequence[Any]] = None, *,
                       openai_client: AsyncOpenAI, cache: Optional[Cache] = None) -> ConsultaResult:
    """Consulta CIMA (chat RAG). `history`: turnos previos de la misma conversación."""
    agent = CIMARagAgent(openai_client, cache=cache)
    try:
        raw = await agent.ask(question, history=_history_dicts(history))
    finally:
        await agent.close()

    references = [_to_reference(ref) for ref in raw.get("references", [])]
    return ConsultaResult(
        answer=raw.get("answer", ""),
        reasoning=raw.get("reasoning", ""),
        references=references,
        success=bool(raw.get("success", False)),
    )


async def run_formulacion(query: str, *, openai_client: AsyncOpenAI, advanced_search: bool = True,
                          cache: Optional[Cache] = None) -> FormulacionResult:
    """Formulación magistral. Las peticiones de prospecto se redirigen sin llamar a OpenAI."""
    agent = FormulationAgent(openai_client, cache=cache)
    agent.use_langgraph = advanced_search
    try:
        if agent.detect_formulation_type(query).get("is_prospecto"):
            return FormulacionResult(answer=PROSPECTO_REDIRECT_MESSAGE, redirect="prospecto")
        try:
            raw = await agent.answer_question(query)
        except Exception as e:
            logger.error(f"Formulación failed: {e}")
            return FormulacionResult(
                answer="Se produjo un error generando la formulación. Inténtelo de nuevo.",
                success=False,
            )
    finally:
        await agent.close()

    answer = raw.get("answer", "")
    return FormulacionResult(
        answer=answer,
        context=raw.get("context", ""),
        references=extract_references(answer),
    )


async def run_prospecto(query: str, *, openai_client: AsyncOpenAI,
                        cache: Optional[Cache] = None) -> ProspectoResult:
    """Prospecto en formato AEMPS a partir del prospecto registrado en CIMA."""
    # ProspectoGenerator usa la caché por defecto a través de MedicationSearchGraph
    generator = ProspectoGenerator(openai_client)
    try:
        raw = await generator.generate_prospecto(query)
    finally:
        await generator.close()

    content = raw.get("prospecto", "")
    context = raw.get("context", "")
    failed = (not context) or content.startswith("Error al generar el prospecto")
    match = _NREGISTRO_PATTERN.search(context)
    medication = raw.get("medication") or None
    return ProspectoResult(
        content=content,
        context=context,
        medication_name=None if medication == "No disponible" else medication,
        nregistro=match.group(1) if match else None,
        success=not failed,
    )
