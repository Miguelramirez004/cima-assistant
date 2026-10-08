# CIMA Assistant API

FastAPI app in `server/`, deployed as the Vercel Python Function `api/index.py`.
Interactive docs: `/api/docs` (OpenAPI at `/api/openapi.json`).

## Authentication and organization

Every endpoint except `/api/health` requires a logged-in Supabase user:

```
Authorization: Bearer <Supabase access token>   # session.access_token from supabase-js
X-Org-Id: <organization uuid>                   # generation endpoints only
```

- Tokens signed with asymmetric keys (ES256/RS256) are verified locally against
  the project's JWKS (cached 10 min); legacy HS256 tokens are checked with
  Supabase Auth (`/auth/v1/user`).
- `X-Org-Id` must be an organization the user belongs to, otherwise `403`.
  The API uses the secret key (bypasses RLS), so this check is the tenant
  boundary on the server; every read and write filters by the verified org
  and user.

| Status | Meaning |
|---|---|
| 400 | Missing or malformed `X-Org-Id` / ids |
| 401 | Not logged in, token invalid or expired |
| 403 | Not a member of the organization / not allowed for the role |
| 404 | Conversation not found (or not yours) |
| 409 | Duplicate (organization slug, pending invitation) |
| 422 | Invalid body (e.g. query shorter than 3 or longer than 2000 chars) |
| 429 | Organization monthly quota exhausted, or more than `USER_REQUESTS_PER_MINUTE` (default 10) requests in the last minute (`Retry-After: 60`) |
| 503 | OpenAI / Supabase not configured or unavailable |

Each request that calls OpenAI writes a `usage_events` row (tokens), which
feeds the monthly quota (`organizations.monthly_request_quota`) and per-org
billing.

## Generation

### `POST /api/formulacion`

```json
{ "query": "Suspensión oral de omeprazol 2 mg/ml", "advanced_search": true }
```

Response: `{ id, answer, context, references: [{title, url, nregistro}], redirect, success }`.
`redirect: "prospecto"` means the query asks for a prospecto: nothing is saved
or charged and the UI should switch to Prospectos. Successful results are saved
to `formulations` and `id` is set.

### `POST /api/prospecto`

```json
{ "query": "Prospecto de ibuprofeno 600 mg" }
```

Response: `{ id, content, context, medication_name, nregistro, success }`.
Saved to `prospectos` when `success`.

### `POST /api/consulta` — Server-Sent Events

```json
{ "question": "¿Contraindicaciones del ibuprofeno?", "conversation_id": null }
```

Omit `conversation_id` to start a conversation (titled with the question);
pass it to continue one (the last 10 messages are sent to the model as
history). Response is `text/event-stream`:

| Event | Data |
|---|---|
| `conversation` | `{ conversation_id, user_message_id }` (always first) |
| `trace` | `{ message }` — one per retrieval step (intent, active principle, medications, sections) |
| `references` | `{ references: [{title, url, nregistro}] }` |
| `token` | `{ text }` — answer fragments as they are generated |
| `done` | `{ answer, reasoning, references, success, message_id }` |
| `error` | `{ detail }` — unexpected failure; no `done` follows |

The user message is stored before generation and the assistant message on
`done`, in `messages`.

## Organizations

### `POST /api/organizations/{org_id}/invitations`

```json
{ "email": "farmaceutica@example.com", "role": "member" }
```

Owners can invite any role; admins `admin`/`member` only. Creates a pending
invitation (7 days) and emails a link to `{APP_URL}/invite/{token}`: a
Supabase invitation email for new users, a magic link for existing ones.
Response: `{ id, email, role, expires_at, invite_url, email_sent }`; if
`email_sent` is false, share `invite_url` manually.

The `/invite/{token}` page (phase 5) signs the user in and calls the
`accept_invitation(token)` RPC.

### `POST /api/admin/organizations` (platform admins)

```json
{ "name": "Farmacia Central", "slug": "farmacia-central",
  "owner_email": "titular@example.com", "monthly_request_quota": 2000 }
```

Creates the organization and an `owner` invitation for `owner_email`.
Response: `{ organization_id, invitation }`.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `SUPABASE_URL` (or `NEXT_PUBLIC_SUPABASE_URL`) | Project URL |
| `SUPABASE_SECRET_KEY` | Server-side database access (never public) |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Used for the HS256 fallback check |
| `APP_URL` | Public URL used in invitation links (defaults to the request origin) |
| `USER_REQUESTS_PER_MINUTE` | Per-user rate limit (default 10) |
| `OPENAI_CHAT_MODEL` | Model (default `gpt-4o-mini`) |

Invitation emails use Supabase's built-in email service, which is heavily
rate-limited (a few emails per hour). Configure custom SMTP in Supabase →
Authentication → Emails before inviting real users.
