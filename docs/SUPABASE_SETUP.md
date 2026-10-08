# Supabase setup

How to create the Supabase project for CIMA Assistant and apply the schema in
`supabase/migrations`. Steps 1–4 are one-off; step 5 repeats for every new
organization.

## What the schema contains

| Migration | Contents |
|---|---|
| `…150000_tenancy.sql` | `organizations`, `profiles`, `organization_members` (roles `owner` / `admin` / `member`), `organization_invitations`; helper functions `is_org_member`, `has_org_role`, `is_platform_admin`; profile-on-signup trigger; "last owner can't leave" guard; `accept_invitation(token)` RPC |
| `…150100_content.sql` | `conversations` + `messages`, `formulations`, `prospectos`, `usage_events`, `cima_cache`; `org_requests_this_month(org)` for quotas |
| `…150200_rls.sql` | Minimal grants (nothing for `anon`) and RLS policies |
| `…150300_cache_cleanup.sql` | `purge_expired_cima_cache()` scheduled nightly with `pg_cron` |

Access rules (all covered by `supabase/tests/database/rls.test.sql`):

- Logged-out visitors can read nothing.
- Users see only organizations they belong to, and only **their own**
  formulations, prospectos and conversations within them.
- Owners and admins see the organization's usage, members and invitations.
  Admins can't invite, promote or remove owners. An organization always keeps
  at least one owner (so a user who is the only owner of an organization
  can't be deleted until ownership is transferred).
- App content, usage and quotas are written only by the API (service role).
- Users join only by accepting an invitation sent to their own email.

## 1. Create the project

1. <https://supabase.com/dashboard> → **New project**.
2. Region: **Central EU (Frankfurt)** — the same area as the Vercel functions
   (`fra1`) and close to AEMPS/CIMA.
3. Save the database password in your password manager.

## 2. Apply the migrations

From your machine, in the repository:

```bash
npm install
npx supabase login
npx supabase link --project-ref <project-ref>   # Settings → General → Project ID
npx supabase db push                           # applies supabase/migrations
```

Then, in the dashboard, check **Database → Extensions** shows `pg_cron`
enabled and **Integrations → Cron** lists `purge-expired-cima-cache`.

## 3. Configure Auth

**Authentication → Sign In / Providers**

- Turn **off** "Allow new users to sign up" (the app is invite-only).
- Email provider: on. Google / Microsoft (Azure) providers: optional, later.

**Authentication → URL Configuration**

- Site URL: your production URL, e.g. `https://cima-assistant.vercel.app`
- Redirect URLs:
  - `https://cima-assistant.vercel.app/**`
  - `https://*-mr0001825-gmailcoms-projects.vercel.app/**` (preview deployments)
  - `http://localhost:3000/**`

## 4. Connect Vercel

Easiest: install the **Supabase** integration from the Vercel Marketplace
and connect it to the `cima-assistant` project; it creates the env vars for
Production, Preview and Development.

Or add them by hand (**Vercel → Project → Settings → Environment Variables**),
from **Supabase → Project Settings → API Keys**:

| Variable | Value | Exposed to the browser |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | Project URL | yes |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | Publishable key (`sb_publishable_…`; the legacy "anon" key also works) | yes — safe, RLS protects the data |
| `SUPABASE_SECRET_KEY` | Secret key (`sb_secret_…`; legacy "service_role") | **never** — server only |

Never prefix the secret key with `NEXT_PUBLIC_`.

## 5. Bootstrap the first admin and organization

1. **Authentication → Users → Invite user** with your email, then accept the
   email invitation.
2. **SQL Editor**, replacing the placeholders:

```sql
-- Make yourself platform admin
update public.profiles set is_platform_admin = true
where id = (select id from auth.users where email = '<your-email>');

-- Create an organization with you as owner
with org as (
  insert into public.organizations (name, slug, monthly_request_quota)
  values ('<Organization name>', '<organization-slug>', 2000)
  returning id
)
insert into public.organization_members (organization_id, user_id, role)
select org.id, u.id, 'owner'
from org, auth.users u
where u.email = '<your-email>';
```

Once the admin pages are built (phase 5) this is done from `/admin`; owners
then invite their own members from the organization settings.

## Testing the schema

```bash
npm run test:db                       # throwaway Postgres + pgTAP (no Docker)
npx supabase start && npx supabase test db   # full local Supabase (Docker)
```

CI (`.github/workflows/ci.yml`) runs the same tests on every pull request.
