"""
Acceso a datos del backend sobre Supabase.

Todas las operaciones usan la clave secreta (saltan RLS), así que cada método
recibe y filtra explícitamente por organización y usuario: es el límite entre
tenants en el lado del servidor. Los tests sustituyen esta clase por una
implementación en memoria con la misma interfaz.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .supabase_rest import SupabaseRest


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


class SupabaseStore:
    def __init__(self, rest: SupabaseRest):
        self.rest = rest

    # ------------------------------------------------------------- tenancy

    async def get_membership_role(self, org_id: str, user_id: str) -> Optional[str]:
        rows = await self.rest.select("organization_members", {
            "select": "role",
            "organization_id": f"eq.{org_id}",
            "user_id": f"eq.{user_id}",
        })
        return rows[0]["role"] if rows else None

    async def is_platform_admin(self, user_id: str) -> bool:
        rows = await self.rest.select("profiles", {
            "select": "is_platform_admin", "id": f"eq.{user_id}",
        })
        return bool(rows and rows[0].get("is_platform_admin"))

    async def get_monthly_quota(self, org_id: str) -> int:
        rows = await self.rest.select("organizations", {
            "select": "monthly_request_quota", "id": f"eq.{org_id}",
        })
        return int(rows[0]["monthly_request_quota"]) if rows else 0

    async def org_requests_this_month(self, org_id: str) -> int:
        return int(await self.rest.rpc("org_requests_this_month", {"org": org_id}))

    async def user_requests_since(self, user_id: str, since: datetime) -> int:
        return await self.rest.count("usage_events", {
            "user_id": f"eq.{user_id}", "created_at": f"gte.{_iso(since)}",
        })

    async def create_organization(self, name: str, slug: str, quota: int) -> Dict[str, Any]:
        rows = await self.rest.insert("organizations", {
            "name": name, "slug": slug, "monthly_request_quota": quota,
        })
        return rows[0]

    async def create_invitation(self, org_id: str, email: str, role: str,
                                invited_by: str) -> Dict[str, Any]:
        rows = await self.rest.insert("organization_invitations", {
            "organization_id": org_id, "email": email, "role": role, "invited_by": invited_by,
        })
        return rows[0]

    # ------------------------------------------------------------- content

    async def record_usage(self, org_id: str, user_id: str, kind: str,
                           prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
        await self.rest.insert("usage_events", {
            "organization_id": org_id, "user_id": user_id, "kind": kind,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        }, returning=False)

    async def save_formulation(self, org_id: str, user_id: str, query: str, answer: str,
                               context: str, references: List[Dict[str, Any]],
                               advanced_search: bool) -> str:
        rows = await self.rest.insert("formulations", {
            "organization_id": org_id, "user_id": user_id, "query": query, "answer": answer,
            "context": context, "references": references, "advanced_search": advanced_search,
        })
        return rows[0]["id"]

    async def save_prospecto(self, org_id: str, user_id: str, query: str, content: str,
                             medication_name: Optional[str], nregistro: Optional[str]) -> str:
        rows = await self.rest.insert("prospectos", {
            "organization_id": org_id, "user_id": user_id, "query": query, "content": content,
            "medication_name": medication_name, "nregistro": nregistro,
        })
        return rows[0]["id"]

    async def get_conversation(self, conversation_id: str, org_id: str,
                               user_id: str) -> Optional[Dict[str, Any]]:
        rows = await self.rest.select("conversations", {
            "select": "id,title",
            "id": f"eq.{conversation_id}",
            "organization_id": f"eq.{org_id}",
            "user_id": f"eq.{user_id}",
        })
        return rows[0] if rows else None

    async def create_conversation(self, org_id: str, user_id: str, title: str) -> str:
        rows = await self.rest.insert("conversations", {
            "organization_id": org_id, "user_id": user_id, "title": title,
        })
        return rows[0]["id"]

    async def recent_messages(self, conversation_id: str, limit: int) -> List[Dict[str, str]]:
        rows = await self.rest.select("messages", {
            "select": "role,content",
            "conversation_id": f"eq.{conversation_id}",
            "order": "created_at.desc",
            "limit": str(limit),
        })
        return list(reversed(rows))

    async def add_message(self, conversation_id: str, org_id: str, user_id: str, role: str,
                          content: str, reasoning: Optional[str] = None,
                          references: Optional[List[Dict[str, Any]]] = None) -> str:
        rows = await self.rest.insert("messages", {
            "conversation_id": conversation_id, "organization_id": org_id, "user_id": user_id,
            "role": role, "content": content, "reasoning": reasoning,
            "references": references or [],
        })
        return rows[0]["id"]


def minute_ago() -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=1)
