# Migration plan: Streamlit → Vercel + Supabase

Status: proposal · Target: CIMA Assistant running on Vercel, with Supabase for
auth, persistence and caching.

---

## 0. Where we are today

| Area | Current state |
|---|---|
| UI | Single Streamlit app (`app.py`, ~1,000 lines): 4 tabs (Formulación, Consultas CIMA, Prospectos, Historial), sidebar toggles, ~350 lines of injected CSS |
| Logic | ~4,000 lines of async Python: `formulacion.py`, `cima_rag.py`, `prospecto.py`, `search_graph.py`, `principle_resolver.py`, `cima_utils.py`, `security.py` |
| External APIs | CIMA REST (`cima.aemps.es/cima/rest`) via `aiohttp`; OpenAI `gpt-4o-mini` via `openai==1.3.0` |
| State | Everything in `st.session_state` — lost on reload, no users, no history across sessions |
| Caching | In-process dicts (`reference_cache`, `ActivePrincipleResolver._cache`) + `@st.cache_resource` |
| Secrets | `.env` / `.streamlit/secrets.toml` (`OPENAI_API_KEY`) |
| Runtime | Python 3.9 (`runtime.txt`), Streamlit Cloud |

### Why Streamlit can't simply be "moved" to Vercel

Streamlit is a long‑lived Python server that keeps a WebSocket open per browser
tab. Vercel runs **serverless functions** (short‑lived, stateless, no persistent
WebSockets). So the migration necessarily means:

1. **A new frontend** (Next.js — Vercel's native framework).
2. **The Python logic exposed as HTTP API endpoints** (Vercel Python Functions /
   FastAPI).
3. **All state that lived in memory moves to Supabase** (history, caches, users).

### Issue the migration fixes

`get_cima_rag_agent()` is decorated with `@st.cache_resource`, so **one
`CIMARagAgent` instance — and its `conversation_history` — is shared by every
user of the deployment**. User A's questions leak into user B's LLM context.
In the new design the conversation is loaded per user from Supabase on each
request, which removes this by construction.

---

## 1. Target architecture

```
Browser ──► Vercel
            ├─ Next.js (App Router, TypeScript, Tailwind)      ← UI
            │    • Supabase Auth (magic link / Google)
            │    • reads history directly from Supabase (RLS)
            │
            └─ /api/*  Python Functions (FastAPI)              ← existing logic
                 • POST /api/formulacion
                 • POST /api/consulta      (streams via SSE)
                 • POST /api/prospecto
                 • verifies Supabase JWT, writes results to Supabase
                 │
                 ├──► CIMA REST API (AEMPS)
                 ├──► OpenAI
                 └──► Supabase Postgres (service role)
                        • profiles, conversations, messages
                        • formulations, prospectos
                        • cima_cache (shared HTTP cache)
                        • usage_events (rate limiting / cost tracking)
```

### Key decisions (recommended)

| Decision | Recommendation | Why |
|---|---|---|
| Frontend | **Next.js 15 App Router + TypeScript + Tailwind + shadcn/ui** | First‑class on Vercel; easy streaming UI; the current "Claude‑style" chat look maps cleanly to Tailwind |
| Backend | **Keep Python, wrap it in FastAPI on Vercel's Python runtime** in the same repo/project | Reuses ~4,000 lines of tested CIMA/RAG logic instead of rewriting. A TypeScript port is optional later (Phase 9) |
| Auth | **Supabase Auth** (email magic link + optional Google) | Gives per‑user history and lets us rate‑limit OpenAI spend |
| DB access | Frontend: `@supabase/ssr` with anon key + RLS. Backend: service‑role key, server‑side only | Users can read only their own rows; only the backend writes AI results |
| Caching | `cima_cache` table in Postgres (TTL) | In‑memory caches die with every cold start on serverless |
| Streaming | Server‑Sent Events from `/api/consulta` | Replaces `st.status` spinner with live trace + token streaming |
| Vector search | **Not needed now** (pgvector available later) | The app retrieves live from the CIMA API, not from an index |

---

## 2. Step‑by‑step plan

Each phase ends in something deployable; the Streamlit app keeps running until
Phase 8.

### Phase 1 — Preparation & repo restructure (≈1 day)

1. Create accounts/projects: Vercel project linked to this GitHub repo; Supabase
   project (choose an **EU region**, e.g. `eu-central-1`/Frankfurt — the users
   and AEMPS are in Spain; keeps latency to CIMA and GDPR posture good).
2. Restructure the repo (on a branch):
   ```
   /                       Next.js app (package.json, app/, components/, lib/)
   /api/index.py           FastAPI entrypoint (Vercel Python function)
   /cima_core/             existing Python modules moved here as a package
       formulacion.py, cima_rag.py, prospecto.py, search_graph.py,
       principle_resolver.py, cima_utils.py, security.py, config.py
   /supabase/migrations/   SQL migrations
   /legacy/app.py          Streamlit UI kept until cut‑over
   requirements.txt        backend deps only
   vercel.json
   ```
3. Upgrade the Python runtime and pins:
   - Python **3.12** (Vercel's Python runtime; drop `runtime.txt`, `packages.txt`, `.devcontainer` Streamlit command).
   - `openai` 1.3.0 → current 1.x (the client API is compatible; re‑test calls).
   - Add `fastapi`, `pydantic>=2.7`, `supabase` (or `httpx` + PostgREST), `PyJWT`.
   - Remove `streamlit` from backend requirements.
4. Install Supabase CLI locally (`supabase init`, `supabase start`) for a local
   Postgres + Auth to develop against.

### Phase 2 — Decouple the Python core from Streamlit (≈1–2 days)

Goal: `cima_core` imports nothing from Streamlit and holds no cross‑request state.

1. **Config**: `config.py` reads only env vars (`OPENAI_API_KEY`,
   `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`). Remove `st.secrets` fallbacks.
2. **Per‑request objects**: agents currently keep `self.session`
   (`aiohttp.ClientSession`) and `conversation_history` on long‑lived instances.
   Change to: create the agent per request, open the `ClientSession` in an
   `async with`, and pass `history: list[dict]` in as an argument to
   `CIMARagAgent.ask(question, history)`.
3. **Pluggable cache**: replace the dict caches
   (`FormulationAgent.reference_cache`, `ProspectoGenerator.reference_cache`,
   `ActivePrincipleResolver._cache`) with a small `Cache` protocol
   (`get(key) / set(key, value, ttl)`). Implementations: `MemoryCache`
   (tests) and `SupabaseCache` (prod).
4. **Remove dead code**: `CIMAExpertAgent` in `formulacion.py` appears unused
   by `app.py` — confirm and delete.
5. **Return structured data**: each entrypoint returns a Pydantic model
   (`FormulationResult`, `ConsultaResult`, `ProspectoResult`) rather than
   loose dicts, so FastAPI generates an OpenAPI schema the frontend can type
   against (`openapi-typescript`).
6. Keep `scripts/validate_principle_resolver.py` working; add `pytest` tests
   for the three entrypoints with mocked CIMA/OpenAI.

### Phase 3 — Supabase schema (≈1 day)

`supabase/migrations/0001_init.sql`:

```sql
-- Profiles (1:1 with auth.users)
create table public.profiles (
  id uuid primary key references auth.users on delete cascade,
  display_name text,
  role text not null default 'user',          -- 'user' | 'admin'
  created_at timestamptz not null default now()
);

-- Consultas CIMA chat
create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  role text not null check (role in ('user','assistant')),
  content text not null,
  reasoning text,                 -- retrieval trace
  references jsonb default '[]',  -- [{nregistro, nombre, url, ...}]
  created_at timestamptz not null default now()
);
create index on public.messages (conversation_id, created_at);

-- Formulación magistral
create table public.formulations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  query text not null,
  answer text not null,
  context text,
  references jsonb default '[]',
  advanced_search boolean not null default true,
  created_at timestamptz not null default now()
);

-- Prospectos
create table public.prospectos (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  query text not null,
  medication_name text,
  nregistro text,
  content text not null,
  created_at timestamptz not null default now()
);

-- Shared CIMA response cache (backend only)
create table public.cima_cache (
  key text primary key,           -- e.g. 'medicamento:12345', 'maestras:paracetamol'
  value jsonb not null,
  expires_at timestamptz not null
);
create index on public.cima_cache (expires_at);

-- Usage / rate limiting
create table public.usage_events (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users on delete cascade,
  kind text not null,             -- 'formulacion' | 'consulta' | 'prospecto'
  prompt_tokens int, completion_tokens int,
  created_at timestamptz not null default now()
);
create index on public.usage_events (user_id, created_at);
```

Row Level Security:

```sql
alter table public.profiles      enable row level security;
alter table public.conversations enable row level security;
alter table public.messages      enable row level security;
alter table public.formulations  enable row level security;
alter table public.prospectos    enable row level security;
alter table public.cima_cache    enable row level security;  -- no policies: service role only
alter table public.usage_events  enable row level security;

-- Users read/delete their own rows; inserts come from the backend (service role)
create policy "own rows" on public.conversations for select using (auth.uid() = user_id);
create policy "own rows del" on public.conversations for delete using (auth.uid() = user_id);
-- repeat select/delete for messages, formulations, prospectos, usage_events
create policy "own profile" on public.profiles for select using (auth.uid() = id);
```

Plus:
- Trigger `on auth.users insert` → create `profiles` row.
- `pg_cron` job (Supabase extension) to `delete from cima_cache where expires_at < now()` nightly.
- Generate TS types: `supabase gen types typescript > lib/database.types.ts`.

### Phase 4 — FastAPI backend on Vercel (≈2–3 days)

1. `api/index.py`:
   - `app = FastAPI()` with routes `POST /api/formulacion`,
     `POST /api/consulta`, `POST /api/prospecto`, `GET /api/health`.
   - Dependency `current_user`: read `Authorization: Bearer <supabase access
     token>`, verify it (Supabase JWT secret / JWKS), return `user_id`.
   - Request bodies validated with Pydantic; reuse `security.clamp_query`.
2. **Consulta (chat)** flow:
   - Load last N messages of `conversation_id` from Supabase (replaces
     in‑memory history → fixes the cross‑user leak).
   - Run the graph; stream SSE events: `trace` (each graph node: analyze →
     resolve → retrieve → sections), `token` (OpenAI streaming), `references`,
     `done`. This needs a small change in `_node_generate` to use
     `stream=True`.
   - Persist user + assistant messages at the end.
3. **Formulación / Prospecto**: same pattern; persist to `formulations` /
   `prospectos`; return the row id. Prospecto‑redirect logic (`"use la pestaña
   'Prospectos'"` string check in `app.py`) becomes a typed field
   `redirect: "prospecto" | null`.
4. **Rate limiting**: before calling OpenAI, count `usage_events` for the user
   in the last hour/day; reject with 429 over the limit. Record tokens after.
5. `vercel.json`:
   ```json
   {
     "functions": {
       "api/index.py": { "maxDuration": 120 }
     },
     "rewrites": [{ "source": "/api/(.*)", "destination": "/api/index.py" }]
   }
   ```
   CIMA + OpenAI calls currently allow up to 60 s timeouts per call; keep
   Fluid Compute on and set `maxDuration` accordingly (check your plan's limit).
   Set the function region to match Supabase (e.g. `fra1`).
6. Local dev: `vercel dev` (runs Next.js and Python functions together) against
   `supabase start`.

### Phase 5 — Next.js frontend (≈4–6 days)

Route map (replaces the 4 Streamlit tabs + sidebar):

| Route | Replaces | Notes |
|---|---|---|
| `/login` | — | Supabase magic link / Google |
| `/formulacion` | Tab 1 | Textarea + examples, "búsqueda avanzada" toggle, result rendered with `react-markdown` + `remark-gfm`, collapsible CIMA context, **Download .md** (client‑side Blob) |
| `/consultas` and `/consultas/[conversationId]` | Tab 2 | Chat UI with streaming (SSE via `fetch` + `ReadableStream`), collapsible "proceso de razonamiento", reference chips linking to ficha técnica, "Nueva conversación" |
| `/prospectos` | Tab 3 | Same pattern as formulación; download |
| `/historial` | Tab 4 + sidebar history | Server component reading the user's rows directly from Supabase (RLS); delete actions |
| Layout sidebar | Sidebar | Recent queries, settings toggles (stored in `localStorage` or `profiles`) |

Implementation notes:
- `middleware.ts` with `@supabase/ssr` to refresh sessions and protect routes.
- `lib/api.ts` wraps calls to `/api/*`, attaching the access token.
- Port the design tokens from the injected CSS in `app.py` (teal `#0D9488`,
  Inter, slate text) into `tailwind.config.ts`.
- Render model output as Markdown **without** raw HTML (`react-markdown`
  default) — keeps the protection `security.escape_html/safe_url` gives today;
  validate reference URLs are `https://cima.aemps.es/...`.
- Disclaimer banner (clinical info comes from CIMA; not medical advice) on every page.

### Phase 6 — Environments, secrets & CI (≈1 day)

Vercel env vars (Production / Preview / Development):

| Variable | Where used | Exposed to browser? |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | frontend | yes |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | frontend | yes (safe with RLS) |
| `SUPABASE_URL` | backend | no |
| `SUPABASE_SERVICE_ROLE_KEY` | backend | **never** |
| `SUPABASE_JWT_SECRET` (or use JWKS) | backend | no |
| `OPENAI_API_KEY` | backend | no |

- Use the **Vercel ↔ Supabase integration** to sync these automatically, and
  Supabase **branching** so each Vercel preview deployment gets its own DB branch.
- GitHub Actions: `ruff` + `pytest` (Python), `tsc` + `eslint` + `next build`
  (frontend), `supabase db lint` / migration check.
- Supabase Auth settings: add production + `*.vercel.app` preview URLs to the
  redirect allow‑list.

### Phase 7 — Testing & hardening (≈2 days)

1. Parity check: run the same set of ~20 queries (formulación, consultas, prospectos) in Streamlit and in the new app; compare references and answers.
2. Playwright E2E: login → each feature → history shows item → download works.
3. Load: confirm cold start + typical latency; verify CIMA cache hit rates in `cima_cache`.
4. Security: RLS tests (user A cannot read user B's rows), service key not in client bundle (`grep` the `.next` output), 429 rate‑limit behavior, CORS same‑origin only.
5. Observability: Vercel logs + Supabase logs; optionally Sentry for both runtimes.

### Phase 8 — Cut‑over (≈½ day)

1. Deploy to production on Vercel; attach custom domain.
2. Point users to the new URL; put a notice/redirect link on the Streamlit app.
3. After a stable period, delete `legacy/`, `.streamlit/`, `packages.txt`,
   `runtime.txt`, Streamlit dev‑container config; update `README.md` and
   `SETUP_GUIDE.md`.
4. No data migration is required — Streamlit history was never persisted.

### Phase 9 — Optional follow‑ups

- **Port the backend to TypeScript** (Next.js Route Handlers + Vercel AI SDK):
  one language, Edge‑friendly streaming, no Python cold starts. Worth it only
  once the Python logic is stable; do it module by module behind the same API
  contract.
- Shared/exportable formulations (public read link via a `share_token`).
- PDF export of prospectos (server‑side).
- pgvector index of fichas técnicas for faster semantic retrieval.
- Admin dashboard over `usage_events` (cost per user/day).

---

## 3. Timeline summary

| Phase | Effort |
|---|---|
| 1. Prep & restructure | 1 day |
| 2. Decouple Python core | 1–2 days |
| 3. Supabase schema + RLS | 1 day |
| 4. FastAPI on Vercel | 2–3 days |
| 5. Next.js frontend | 4–6 days |
| 6. Env, secrets, CI | 1 day |
| 7. Testing & hardening | 2 days |
| 8. Cut‑over | ½ day |
| **Total** | **≈ 2.5–3.5 weeks** for one developer |

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Function timeouts on slow CIMA responses | Postgres cache, parallel requests (already `asyncio.gather`), `maxDuration`, stream partial progress |
| Python cold starts | Keep deps lean (no Streamlit/tiktoken in hot path if possible), Fluid Compute |
| OpenAI cost abuse once public | Auth required + `usage_events` rate limits |
| CIMA API rate limits / outages | Cache with TTL, graceful error messages (already present), retry with backoff |
| Leaking service role key | Only in server env vars; CI check on client bundle |
| Health‑data/GDPR concerns | EU region, users can delete their history, no PII in prompts beyond the query |

## 5. Open questions to decide before starting

1. Should the app require login, or allow anonymous use (Supabase anonymous sign‑ins) with history only for signed‑in users?
2. Keep Python backend (recommended for speed of migration) or go straight to a full TypeScript rewrite?
3. Who are the users (single pharmacy, multiple organizations)? If multiple orgs, add an `organizations` table and org‑scoped RLS now.
4. Vercel plan (Hobby vs Pro) — affects `maxDuration` and commercial use.
