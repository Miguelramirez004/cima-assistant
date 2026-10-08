"""Endpoints de la API (/api/*)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from cima_core import (
    FormulacionResult, ProspectoResult, TrackedOpenAI, run_formulacion, run_prospecto,
    stream_consulta,
)
from cima_core.cima_rag import MAX_HISTORY_TURNS

from .auth import AuthUser
from .deps import (
    Backend, OrgContext, current_org, current_user, enforce_quota, get_backend, membership,
    parse_uuid, require_store,
)
from .store import SupabaseStore
from .supabase_rest import SupabaseError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

QueryText = Field(min_length=3, max_length=2000)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------- modelos

class FormulacionRequest(BaseModel):
    query: str = QueryText
    advanced_search: bool = True


class FormulacionResponse(FormulacionResult):
    id: Optional[str] = None


class ProspectoRequest(BaseModel):
    query: str = QueryText


class ProspectoResponse(ProspectoResult):
    id: Optional[str] = None


class ConsultaRequest(BaseModel):
    question: str = QueryText
    conversation_id: Optional[str] = None


def _normalize_email(value: str) -> str:
    value = value.strip().lower()
    if not EMAIL_RE.match(value):
        raise ValueError("Email no válido")
    return value


class InvitationRequest(BaseModel):
    email: str = Field(max_length=254)
    role: Literal["owner", "admin", "member"] = "member"

    _email = field_validator("email")(_normalize_email)


class InvitationResponse(BaseModel):
    id: str
    email: str
    role: str
    expires_at: str
    invite_url: str
    email_sent: bool


class CreateOrganizationRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=60)
    owner_email: str = Field(max_length=254)
    monthly_request_quota: int = Field(default=2000, ge=0)

    _owner_email = field_validator("owner_email")(_normalize_email)


class CreateOrganizationResponse(BaseModel):
    organization_id: str
    invitation: InvitationResponse


# --------------------------------------------------------------- generación

@router.post("/formulacion", response_model=FormulacionResponse)
async def formulacion(body: FormulacionRequest, ctx: OrgContext = Depends(current_org),
                      store: SupabaseStore = Depends(require_store),
                      backend: Backend = Depends(get_backend)) -> FormulacionResponse:
    await enforce_quota(ctx, store, backend.settings)
    client = TrackedOpenAI(backend.openai_client())
    result = await run_formulacion(body.query, openai_client=client,
                                   advanced_search=body.advanced_search, cache=backend.cache)
    await _record_usage(store, ctx, "formulacion", client)

    response = FormulacionResponse(**result.model_dump())
    if result.success and result.redirect is None:
        response.id = await store.save_formulation(
            ctx.org_id, ctx.user.id, body.query, result.answer, result.context,
            [r.model_dump() for r in result.references], body.advanced_search,
        )
    return response


@router.post("/prospecto", response_model=ProspectoResponse)
async def prospecto(body: ProspectoRequest, ctx: OrgContext = Depends(current_org),
                    store: SupabaseStore = Depends(require_store),
                    backend: Backend = Depends(get_backend)) -> ProspectoResponse:
    await enforce_quota(ctx, store, backend.settings)
    client = TrackedOpenAI(backend.openai_client())
    result = await run_prospecto(body.query, openai_client=client, cache=backend.cache)
    await _record_usage(store, ctx, "prospecto", client)

    response = ProspectoResponse(**result.model_dump())
    if result.success:
        response.id = await store.save_prospecto(
            ctx.org_id, ctx.user.id, body.query, result.content,
            result.medication_name, result.nregistro,
        )
    return response


@router.post("/consulta")
async def consulta(body: ConsultaRequest, ctx: OrgContext = Depends(current_org),
                   store: SupabaseStore = Depends(require_store),
                   backend: Backend = Depends(get_backend)) -> StreamingResponse:
    """
    Consulta CIMA en streaming (Server-Sent Events). Eventos, en orden:
    `conversation` {conversation_id, user_message_id}, `trace` {message}*,
    `references` {references}, `token` {text}*, `done` {answer, reasoning,
    references, success, message_id}. Un fallo inesperado emite `error`.
    """
    await enforce_quota(ctx, store, backend.settings)
    openai_client = backend.openai_client()
    if body.conversation_id:
        conversation_id = parse_uuid(body.conversation_id, "conversation_id")
        if await store.get_conversation(conversation_id, ctx.org_id, ctx.user.id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversación no encontrada")
        history = await store.recent_messages(conversation_id, MAX_HISTORY_TURNS * 2)
    else:
        conversation_id = await store.create_conversation(ctx.org_id, ctx.user.id, _title(body.question))
        history = []

    user_message_id = await store.add_message(conversation_id, ctx.org_id, ctx.user.id,
                                              "user", body.question)

    async def events() -> AsyncIterator[str]:
        yield _sse("conversation", {"conversation_id": conversation_id,
                                    "user_message_id": user_message_id})
        try:
            final: Optional[Dict[str, Any]] = None
            async for event in stream_consulta(body.question, history,
                                               openai_client=openai_client, cache=backend.cache):
                if event["type"] == "done":
                    final = event
                else:
                    yield _sse(event["type"], {k: v for k, v in event.items() if k != "type"})
            if final is None:
                raise RuntimeError("El flujo de consulta terminó sin respuesta")

            usage = final.get("usage")
            if usage is not None:
                await store.record_usage(ctx.org_id, ctx.user.id, "consulta",
                                         usage["prompt_tokens"], usage["completion_tokens"])
            message_id = await store.add_message(
                conversation_id, ctx.org_id, ctx.user.id, "assistant", final["answer"],
                reasoning=final["reasoning"], references=final["references"],
            )
            yield _sse("done", {
                "answer": final["answer"], "reasoning": final["reasoning"],
                "references": final["references"], "success": final["success"],
                "message_id": message_id,
            })
        except Exception as e:
            logger.exception(f"consulta stream failed: {e}")
            yield _sse("error", {"detail": "Se produjo un error procesando la consulta"})

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    })


# ------------------------------------------------------------- organizaciones

@router.post("/organizations/{org_id}/invitations", response_model=InvitationResponse,
             status_code=status.HTTP_201_CREATED)
async def invite_member(org_id: str, body: InvitationRequest, request: Request,
                        user: AuthUser = Depends(current_user),
                        store: SupabaseStore = Depends(require_store),
                        backend: Backend = Depends(get_backend)) -> InvitationResponse:
    """Invita a un usuario a la organización (owners: cualquier rol; admins: no owners)."""
    ctx = await membership(parse_uuid(org_id, "Organización"), user, store)
    if ctx.role not in ("owner", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo owners y admins pueden invitar")
    if body.role == "owner" and ctx.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo un owner puede invitar a otro owner")
    return await _invite(store, backend, request, ctx.org_id, body.email, body.role, user.id)


@router.post("/admin/organizations", response_model=CreateOrganizationResponse,
             status_code=status.HTTP_201_CREATED)
async def create_organization(body: CreateOrganizationRequest, request: Request,
                              user: AuthUser = Depends(current_user),
                              store: SupabaseStore = Depends(require_store),
                              backend: Backend = Depends(get_backend)) -> CreateOrganizationResponse:
    """Alta de una organización e invitación a su primer owner (solo administradores de plataforma)."""
    if not await store.is_platform_admin(user.id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Solo administradores de la plataforma")
    try:
        org = await store.create_organization(body.name, body.slug, body.monthly_request_quota)
    except SupabaseError as e:
        if e.status == status.HTTP_409_CONFLICT:
            raise HTTPException(status.HTTP_409_CONFLICT, "Ya existe una organización con ese identificador")
        raise
    invitation = await _invite(store, backend, request, org["id"], body.owner_email, "owner", user.id)
    return CreateOrganizationResponse(organization_id=org["id"], invitation=invitation)


# ------------------------------------------------------------------ helpers

async def _invite(store: SupabaseStore, backend: Backend, request: Request, org_id: str,
                  email: str, role: str, invited_by: str) -> InvitationResponse:
    try:
        invitation = await store.create_invitation(org_id, email, role, invited_by)
    except SupabaseError as e:
        if e.status == status.HTTP_409_CONFLICT:
            raise HTTPException(status.HTTP_409_CONFLICT, "Ya hay una invitación pendiente para ese email")
        raise

    invite_url = f"{backend.app_url(request)}/invite/{invitation['token']}"
    email_sent = True
    try:
        await store.rest.invite_user(email, invite_url)
    except SupabaseError as e:
        # Usuario ya registrado: enviarle un enlace de acceso que lleva a la invitación
        try:
            if e.status != 422:
                raise
            await store.rest.send_magic_link(email, invite_url)
        except SupabaseError as e2:
            logger.warning(f"Invitation email to {email} failed: {e2}")
            email_sent = False

    return InvitationResponse(
        id=invitation["id"], email=email, role=role, expires_at=invitation["expires_at"],
        invite_url=invite_url, email_sent=email_sent,
    )


async def _record_usage(store: SupabaseStore, ctx: OrgContext, kind: str, client: TrackedOpenAI) -> None:
    if client.calls:
        await store.record_usage(ctx.org_id, ctx.user.id, kind,
                                 client.prompt_tokens, client.completion_tokens)


def _title(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 80 else title[:77].rstrip() + "..."


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
