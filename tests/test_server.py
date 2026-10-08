"""Endpoints de la API con un almacén en memoria (sin Supabase ni OpenAI reales)."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import server.routes as routes
from cima_core import FormulacionResult, ProspectoResult
from cima_core.cache import MemoryCache
from server.app import app
from server.auth import AuthError, AuthUser
from server.config import Settings
from server.deps import Backend, get_backend
from server.supabase_rest import SupabaseError
from tests.conftest import FakeOpenAI

ORG_A = "aaaaaaaa-0000-0000-0000-000000000000"
ORG_B = "bbbbbbbb-0000-0000-0000-000000000000"
OWNER, ADMIN, MEMBER, OUTSIDER = "u-owner", "u-admin", "u-member", "u-outsider"


class FakeAuthRest:
    """Parte de SupabaseRest que usan las invitaciones."""

    def __init__(self):
        self.invited: List[str] = []
        self.magic_links: List[str] = []
        self.existing_users = set()
        self.email_down = False

    async def invite_user(self, email, redirect_to):
        if self.email_down:
            raise SupabaseError(500, "smtp down")
        if email in self.existing_users:
            raise SupabaseError(422, "A user with this email address has already been registered")
        self.invited.append(redirect_to)

    async def send_magic_link(self, email, redirect_to):
        self.magic_links.append(redirect_to)


@dataclass
class FakeStore:
    rest: FakeAuthRest = field(default_factory=FakeAuthRest)
    members: Dict[tuple, str] = field(default_factory=lambda: {
        (ORG_A, OWNER): "owner", (ORG_A, ADMIN): "admin", (ORG_A, MEMBER): "member",
    })
    quota: int = 100
    org_month_requests: int = 0
    user_minute_requests: int = 0
    platform_admins: set = field(default_factory=set)
    usage: List[Dict[str, Any]] = field(default_factory=list)
    formulations: List[Dict[str, Any]] = field(default_factory=list)
    prospectos: List[Dict[str, Any]] = field(default_factory=list)
    conversations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    invitations: List[Dict[str, Any]] = field(default_factory=list)
    organizations: List[Dict[str, Any]] = field(default_factory=list)

    async def get_membership_role(self, org_id, user_id):
        return self.members.get((org_id, user_id))

    async def is_platform_admin(self, user_id):
        return user_id in self.platform_admins

    async def get_monthly_quota(self, org_id):
        return self.quota

    async def org_requests_this_month(self, org_id):
        return self.org_month_requests

    async def user_requests_since(self, user_id, since):
        return self.user_minute_requests

    async def record_usage(self, org_id, user_id, kind, prompt_tokens, completion_tokens):
        self.usage.append(dict(org=org_id, user=user_id, kind=kind,
                               prompt=prompt_tokens, completion=completion_tokens))

    async def save_formulation(self, org_id, user_id, query, answer, context, references, advanced_search):
        self.formulations.append(dict(org=org_id, user=user_id, query=query, references=references))
        return "f-1"

    async def save_prospecto(self, org_id, user_id, query, content, medication_name, nregistro):
        self.prospectos.append(dict(org=org_id, user=user_id, nregistro=nregistro))
        return "p-1"

    async def get_conversation(self, conversation_id, org_id, user_id):
        conv = self.conversations.get(conversation_id)
        if conv and conv["org"] == org_id and conv["user"] == user_id:
            return {"id": conversation_id, "title": conv["title"]}
        return None

    async def create_conversation(self, org_id, user_id, title):
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = dict(org=org_id, user=user_id, title=title)
        return conversation_id

    async def recent_messages(self, conversation_id, limit):
        rows = [{"role": m["role"], "content": m["content"]}
                for m in self.messages if m["conversation"] == conversation_id]
        return rows[-limit:]

    async def add_message(self, conversation_id, org_id, user_id, role, content,
                          reasoning=None, references=None):
        self.messages.append(dict(conversation=conversation_id, org=org_id, user=user_id,
                                  role=role, content=content, references=references))
        return f"m-{len(self.messages)}"

    async def create_organization(self, name, slug, quota):
        if any(o["slug"] == slug for o in self.organizations):
            raise SupabaseError(409, "duplicate key")
        org = {"id": str(uuid.uuid4()), "name": name, "slug": slug}
        self.organizations.append(org)
        return org

    async def create_invitation(self, org_id, email, role, invited_by):
        if any(i["org"] == org_id and i["email"] == email for i in self.invitations):
            raise SupabaseError(409, "duplicate key")
        invitation = {"id": str(uuid.uuid4()), "org": org_id, "email": email, "role": role,
                      "token": "tok123", "expires_at": "2026-10-15T00:00:00+00:00"}
        self.invitations.append(invitation)
        return invitation


class FakeVerifier:
    async def verify(self, token):
        if not token.startswith("tok-"):
            raise AuthError("Token no válido o caducado")
        return AuthUser(id=token[4:], email=f"{token[4:]}@example.test")


class FakeBackend(Backend):
    def __init__(self, store, openai):
        settings = Settings(supabase_url="https://x.supabase.co", supabase_publishable_key="pk",
                            supabase_secret_key="sk", openai_api_key="sk-test",
                            app_url="https://app.test", user_requests_per_minute=10)
        super().__init__(settings=settings, store=store, verifier=FakeVerifier(), cache=MemoryCache())
        self._openai = openai

    def openai_client(self):
        return self._openai


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def openai():
    return FakeOpenAI()


@pytest.fixture
def client(store, openai):
    app.dependency_overrides[get_backend] = lambda: FakeBackend(store, openai)
    yield TestClient(app)
    app.dependency_overrides.clear()


def headers(user=MEMBER, org=ORG_A):
    h = {"Authorization": f"Bearer tok-{user}"}
    if org:
        h["X-Org-Id"] = org
    return h


def usage_reply(answer="ok", prompt=100, completion=20):
    """FakeOpenAI cuya respuesta incluye usage, como la API real."""
    from types import SimpleNamespace
    fake = FakeOpenAI(answer)
    create = fake.chat.completions.create

    async def create_with_usage(**kwargs):
        response = await create(**kwargs)
        response.usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
        return response

    fake.chat.completions.create = create_with_usage
    return fake


# ------------------------------------------------------------------ acceso

def test_requires_login(client):
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"})
    assert response.status_code == 401


def test_rejects_invalid_token(client):
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"},
                           headers={"Authorization": "Bearer forged", "X-Org-Id": ORG_A})
    assert response.status_code == 401


def test_requires_org_header(client):
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"},
                           headers=headers(org=None))
    assert response.status_code == 400


def test_rejects_malformed_org_header(client):
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"},
                           headers=headers(org="not-a-uuid"))
    assert response.status_code == 400


def test_rejects_org_the_user_does_not_belong_to(client, store):
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"},
                           headers=headers(org=ORG_B))
    assert response.status_code == 403
    assert store.formulations == [] and store.usage == []


# ------------------------------------------------------------------ cuotas

def test_monthly_quota_exhausted(client, store):
    store.org_month_requests = store.quota
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol"}, headers=headers())
    assert response.status_code == 429
    assert "cuota mensual" in response.json()["detail"]


def test_per_minute_rate_limit(client, store):
    store.user_minute_requests = 10
    response = client.post("/api/prospecto", json={"query": "Prospecto de ibuprofeno"}, headers=headers())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"


def test_quota_rejection_creates_no_conversation(client, store):
    store.org_month_requests = store.quota
    response = client.post("/api/consulta", json={"question": "¿Dosis de ibuprofeno?"}, headers=headers())
    assert response.status_code == 429
    assert store.conversations == {} and store.messages == []


# --------------------------------------------------------------- formulación

def test_formulacion_saves_result_and_usage(client, store, monkeypatch):
    fake_openai = usage_reply(prompt=1200, completion=300)
    app.dependency_overrides[get_backend] = lambda: FakeBackend(store, fake_openai)

    async def fake_run(query, *, openai_client, advanced_search, cache):
        await openai_client.chat.completions.create(model="m", messages=[])
        return FormulacionResult(answer="Fórmula", context="ctx",
                                 references=[{"title": "X", "url": "https://u", "nregistro": "1"}])

    monkeypatch.setattr(routes, "run_formulacion", fake_run)
    response = client.post("/api/formulacion", json={"query": "Suspensión de omeprazol 2 mg/ml"},
                           headers=headers())
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "f-1" and body["answer"] == "Fórmula"
    assert store.formulations[0]["org"] == ORG_A and store.formulations[0]["user"] == MEMBER
    assert store.usage == [dict(org=ORG_A, user=MEMBER, kind="formulacion", prompt=1200, completion=300)]


def test_formulacion_prospecto_redirect_is_not_saved_or_charged(client, store, monkeypatch):
    async def fake_run(query, *, openai_client, advanced_search, cache):
        return FormulacionResult(answer="Use Prospectos", redirect="prospecto")

    monkeypatch.setattr(routes, "run_formulacion", fake_run)
    response = client.post("/api/formulacion", json={"query": "Redactar un prospecto"}, headers=headers())
    assert response.json()["redirect"] == "prospecto"
    assert response.json()["id"] is None
    assert store.formulations == [] and store.usage == []


def test_query_length_is_validated(client):
    response = client.post("/api/formulacion", json={"query": "x" * 2001}, headers=headers())
    assert response.status_code == 422


# ---------------------------------------------------------------- prospecto

def test_prospecto_saved(client, store, monkeypatch):
    async def fake_run(query, *, openai_client, cache):
        return ProspectoResult(content="PROSPECTO", context="ctx", medication_name="DALSY", nregistro="67890")

    monkeypatch.setattr(routes, "run_prospecto", fake_run)
    response = client.post("/api/prospecto", json={"query": "Prospecto de ibuprofeno"}, headers=headers())
    assert response.status_code == 200 and response.json()["id"] == "p-1"
    assert store.prospectos == [dict(org=ORG_A, user=MEMBER, nregistro="67890")]


def test_failed_prospecto_not_saved(client, store, monkeypatch):
    async def fake_run(query, *, openai_client, cache):
        return ProspectoResult(content="Error al generar el prospecto: x", success=False)

    monkeypatch.setattr(routes, "run_prospecto", fake_run)
    response = client.post("/api/prospecto", json={"query": "Prospecto de ibuprofeno"}, headers=headers())
    assert response.json()["success"] is False and response.json()["id"] is None
    assert store.prospectos == []


# ----------------------------------------------------------------- consulta

def parse_sse(text: str) -> List[tuple]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def fake_stream(seen: Dict[str, Any]):
    async def stream(question, history, *, openai_client, cache):
        seen["history"] = history
        yield {"type": "trace", "message": "Intención detectada"}
        yield {"type": "references", "references": [{"title": "DALSY", "url": "u", "nregistro": "1"}]}
        yield {"type": "token", "text": "Hola "}
        yield {"type": "token", "text": "mundo"}
        yield {"type": "done", "answer": "Hola mundo", "reasoning": "• paso", "success": True,
               "references": [{"title": "DALSY", "url": "u", "nregistro": "1"}],
               "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
    return stream


def test_consulta_streams_and_persists(client, store, monkeypatch):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(routes, "stream_consulta", fake_stream(seen))
    response = client.post("/api/consulta", json={"question": "¿Dosis de ibuprofeno?"}, headers=headers())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [e for e, _ in events] == ["conversation", "trace", "references", "token", "token", "done"]
    conversation_id = events[0][1]["conversation_id"]
    assert store.conversations[conversation_id]["title"] == "¿Dosis de ibuprofeno?"
    assert events[-1][1]["answer"] == "Hola mundo" and events[-1][1]["message_id"]
    assert [m["role"] for m in store.messages] == ["user", "assistant"]
    assert store.usage == [dict(org=ORG_A, user=MEMBER, kind="consulta", prompt=50, completion=5)]
    assert seen["history"] == []


def test_consulta_continues_conversation_with_history(client, store, monkeypatch):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(routes, "stream_consulta", fake_stream(seen))
    first = parse_sse(client.post("/api/consulta", json={"question": "¿Dosis de ibuprofeno?"},
                                  headers=headers()).text)
    conversation_id = first[0][1]["conversation_id"]

    client.post("/api/consulta", json={"question": "¿Y en niños?", "conversation_id": conversation_id},
                headers=headers())
    assert seen["history"] == [
        {"role": "user", "content": "¿Dosis de ibuprofeno?"},
        {"role": "assistant", "content": "Hola mundo"},
    ]
    assert len(store.conversations) == 1


def test_consulta_cannot_use_someone_elses_conversation(client, store, monkeypatch):
    monkeypatch.setattr(routes, "stream_consulta", fake_stream({}))
    first = parse_sse(client.post("/api/consulta", json={"question": "¿Dosis de ibuprofeno?"},
                                  headers=headers(user=OWNER)).text)
    response = client.post("/api/consulta", headers=headers(user=MEMBER),
                           json={"question": "¿Y en niños?", "conversation_id": first[0][1]["conversation_id"]})
    assert response.status_code == 404


def test_consulta_stream_error_event(client, store, monkeypatch):
    async def broken(question, history, *, openai_client, cache):
        yield {"type": "trace", "message": "x"}
        raise RuntimeError("boom")

    monkeypatch.setattr(routes, "stream_consulta", broken)
    events = parse_sse(client.post("/api/consulta", json={"question": "¿Dosis de ibuprofeno?"},
                                   headers=headers()).text)
    assert events[-1][0] == "error"


# ------------------------------------------------------------- invitaciones

def invite(client, user, role="member", email="New.User@Example.com", org=ORG_A):
    return client.post(f"/api/organizations/{org}/invitations", json={"email": email, "role": role},
                       headers=headers(user=user, org=None))


def test_member_cannot_invite(client):
    assert invite(client, MEMBER).status_code == 403


def test_admin_cannot_invite_owner(client):
    assert invite(client, ADMIN, role="owner").status_code == 403


def test_outsider_cannot_invite(client):
    assert invite(client, OUTSIDER).status_code == 403


def test_admin_invites_member_and_email_is_sent(client, store):
    response = invite(client, ADMIN)
    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new.user@example.com" and body["email_sent"] is True
    assert body["invite_url"] == "https://app.test/invite/tok123"
    assert store.rest.invited == ["https://app.test/invite/tok123"]


def test_existing_user_gets_magic_link(client, store):
    store.rest.existing_users.add("new.user@example.com")
    response = invite(client, OWNER, role="owner")
    assert response.status_code == 201 and response.json()["email_sent"] is True
    assert store.rest.magic_links == ["https://app.test/invite/tok123"]


def test_email_failure_still_returns_link(client, store):
    store.rest.email_down = True
    response = invite(client, OWNER)
    assert response.status_code == 201 and response.json()["email_sent"] is False


def test_duplicate_pending_invitation(client):
    assert invite(client, OWNER).status_code == 201
    assert invite(client, OWNER).status_code == 409


def test_invalid_email_rejected(client):
    assert invite(client, OWNER, email="not-an-email").status_code == 422


# ----------------------------------------------------- organizaciones (admin)

def create_org(client, user):
    return client.post("/api/admin/organizations", headers=headers(user=user, org=None), json={
        "name": "Farmacia Nueva", "slug": "farmacia-nueva", "owner_email": "jefa@farmacia.test",
    })


def test_only_platform_admins_create_organizations(client):
    assert create_org(client, OWNER).status_code == 403


def test_platform_admin_creates_org_and_owner_invitation(client, store):
    store.platform_admins.add(OWNER)
    response = create_org(client, OWNER)
    assert response.status_code == 201
    assert store.invitations[0]["role"] == "owner"
    assert store.invitations[0]["org"] == response.json()["organization_id"]
    assert create_org(client, OWNER).status_code == 409


# ------------------------------------------------------------------- health

def test_health_reports_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["supabase_configured"] is True and body["openai_configured"] is True
