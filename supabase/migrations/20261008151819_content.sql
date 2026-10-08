-- =============================================================================
-- Content produced by the app. Every row carries organization_id + user_id.
-- Rows are written by the API (service role) after each request; users read
-- and delete their own rows through RLS.
-- =============================================================================

-- Consultas CIMA (chat)
create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  title text check (char_length(title) <= 200),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  -- Target of the composite FK below: a message always belongs to a
  -- conversation of the same organization and user
  unique (id, organization_id, user_id)
);
create index conversations_org_user_updated_idx
  on public.conversations (organization_id, user_id, updated_at desc);
create index conversations_user_id_idx on public.conversations (user_id);

create table public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null,
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  reasoning text,                                  -- retrieval trace
  "references" jsonb not null default '[]'::jsonb, -- [{title, url, nregistro}]
  created_at timestamptz not null default now(),
  foreign key (conversation_id, organization_id, user_id)
    references public.conversations (id, organization_id, user_id) on delete cascade
);
create index messages_conversation_created_idx on public.messages (conversation_id, created_at);
create index messages_conv_org_user_idx on public.messages (conversation_id, organization_id, user_id);
create index messages_organization_id_idx on public.messages (organization_id);
create index messages_user_id_idx on public.messages (user_id);

-- Formulación magistral
create table public.formulations (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  query text not null,
  answer text not null,
  context text,
  "references" jsonb not null default '[]'::jsonb,
  advanced_search boolean not null default true,
  created_at timestamptz not null default now()
);
create index formulations_org_user_created_idx
  on public.formulations (organization_id, user_id, created_at desc);
create index formulations_user_id_idx on public.formulations (user_id);

-- Prospectos
create table public.prospectos (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  query text not null,
  medication_name text,
  nregistro text,
  content text not null,
  created_at timestamptz not null default now()
);
create index prospectos_org_user_created_idx
  on public.prospectos (organization_id, user_id, created_at desc);
create index prospectos_user_id_idx on public.prospectos (user_id);

-- OpenAI usage per request: quotas, rate limiting and per-organization billing
create table public.usage_events (
  id bigint generated always as identity primary key,
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  kind text not null check (kind in ('formulacion', 'consulta', 'prospecto')),
  prompt_tokens integer,
  completion_tokens integer,
  created_at timestamptz not null default now()
);
create index usage_events_org_created_idx on public.usage_events (organization_id, created_at);
create index usage_events_user_created_idx on public.usage_events (user_id, created_at);

-- Shared cache of CIMA API responses (public data, not tenant data).
-- Only the API (service role) reads and writes it.
create table public.cima_cache (
  key text primary key,            -- e.g. 'maestras:1:ibuprofeno', 'medicamento_detalle:12345'
  value jsonb not null,
  expires_at timestamptz not null
);
create index cima_cache_expires_at_idx on public.cima_cache (expires_at);

-- Keep conversations.updated_at current when a message is added
create function public.touch_conversation()
returns trigger
language plpgsql security definer set search_path = ''
as $$
begin
  update public.conversations set updated_at = now() where id = new.conversation_id;
  return new;
end;
$$;

create trigger messages_touch_conversation
  after insert on public.messages
  for each row execute function public.touch_conversation();

-- Requests this calendar month (UTC) for an organization. Used by the API
-- (service role) to enforce monthly_request_quota.
create function public.org_requests_this_month(org uuid)
returns bigint
language sql stable security definer set search_path = ''
as $$
  select count(*) from public.usage_events e
  where e.organization_id = org
    and e.created_at >= date_trunc('month', now() at time zone 'utc') at time zone 'utc';
$$;

revoke execute on function public.org_requests_this_month(uuid) from public, anon, authenticated;
grant execute on function public.org_requests_this_month(uuid) to service_role;
