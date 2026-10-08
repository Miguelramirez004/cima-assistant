-- =============================================================================
-- Tenancy: organizations, memberships, invitations and user profiles.
--
-- Model: shared database; every tenant row carries organization_id and is
-- protected by RLS (see the *_rls.sql migration). Sign-up is disabled in Auth;
-- users join an organization through an invitation.
-- =============================================================================

create type public.org_role as enum ('owner', 'admin', 'member');

create table public.organizations (
  id uuid primary key default gen_random_uuid(),
  name text not null check (char_length(name) between 2 and 120),
  slug text not null unique check (slug ~ '^[a-z0-9]+(-[a-z0-9]+)*$' and char_length(slug) <= 60),
  -- OpenAI-backed requests allowed per calendar month (enforced by the API)
  monthly_request_quota integer not null default 2000 check (monthly_request_quota >= 0),
  created_at timestamptz not null default now()
);

create table public.profiles (
  id uuid primary key references auth.users on delete cascade,
  display_name text check (char_length(display_name) <= 120),
  -- Platform operators: create organizations and set quotas (/admin)
  is_platform_admin boolean not null default false,
  -- Organization selected in the UI switcher
  last_organization_id uuid references public.organizations on delete set null,
  created_at timestamptz not null default now()
);
create index profiles_last_organization_id_idx on public.profiles (last_organization_id);

create table public.organization_members (
  organization_id uuid not null references public.organizations on delete cascade,
  user_id uuid not null references auth.users on delete cascade,
  role public.org_role not null default 'member',
  created_at timestamptz not null default now(),
  primary key (organization_id, user_id)
);
create index organization_members_user_id_idx on public.organization_members (user_id);

create table public.organization_invitations (
  id uuid primary key default gen_random_uuid(),
  organization_id uuid not null references public.organizations on delete cascade,
  email text not null check (email = lower(email) and position('@' in email) > 1),
  role public.org_role not null default 'member',
  token text not null unique
    default replace(gen_random_uuid()::text || gen_random_uuid()::text, '-', ''),
  invited_by uuid references auth.users on delete set null,
  expires_at timestamptz not null default now() + interval '7 days',
  accepted_at timestamptz,
  created_at timestamptz not null default now()
);
create index organization_invitations_organization_id_idx on public.organization_invitations (organization_id);
create index organization_invitations_invited_by_idx on public.organization_invitations (invited_by);
-- One pending invitation per email and organization
create unique index organization_invitations_pending_idx
  on public.organization_invitations (organization_id, email) where accepted_at is null;

-- -----------------------------------------------------------------------------
-- Helper functions for RLS. SECURITY DEFINER so policies on
-- organization_members can call them without recursing into its own RLS.
-- -----------------------------------------------------------------------------

create function public.is_org_member(org uuid)
returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (
    select 1 from public.organization_members m
    where m.organization_id = org and m.user_id = (select auth.uid())
  );
$$;

create function public.has_org_role(org uuid, roles public.org_role[])
returns boolean
language sql stable security definer set search_path = ''
as $$
  select exists (
    select 1 from public.organization_members m
    where m.organization_id = org and m.user_id = (select auth.uid()) and m.role = any (roles)
  );
$$;

create function public.is_platform_admin()
returns boolean
language sql stable security definer set search_path = ''
as $$
  select coalesce(
    (select p.is_platform_admin from public.profiles p where p.id = (select auth.uid())),
    false
  );
$$;

-- -----------------------------------------------------------------------------
-- Profile row for every new auth user
-- -----------------------------------------------------------------------------

create function public.handle_new_user()
returns trigger
language plpgsql security definer set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'full_name', new.raw_user_meta_data ->> 'name'))
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- -----------------------------------------------------------------------------
-- An organization must always keep at least one owner. Cascading deletes of
-- the organization itself are allowed.
-- -----------------------------------------------------------------------------

create function public.ensure_org_has_owner()
returns trigger
language plpgsql security definer set search_path = ''
as $$
begin
  if old.role = 'owner'
     and (tg_op = 'DELETE' or new.role <> 'owner' or new.organization_id <> old.organization_id)
     and exists (select 1 from public.organizations o where o.id = old.organization_id)
     and not exists (
       select 1 from public.organization_members m
       where m.organization_id = old.organization_id and m.role = 'owner'
         and m.user_id <> old.user_id
     )
  then
    raise exception 'An organization must keep at least one owner'
      using errcode = 'check_violation';
  end if;
  return coalesce(new, old);
end;
$$;

create trigger organization_members_keep_owner
  before update or delete on public.organization_members
  for each row execute function public.ensure_org_has_owner();

-- -----------------------------------------------------------------------------
-- Invitation acceptance. The invited user calls this RPC after logging in;
-- the email of the logged-in user must match the invitation.
-- -----------------------------------------------------------------------------

create function public.accept_invitation(invite_token text)
returns uuid
language plpgsql security definer set search_path = ''
as $$
declare
  uid uuid := (select auth.uid());
  user_email text;
  invite public.organization_invitations%rowtype;
begin
  if uid is null then
    raise exception 'Not authenticated' using errcode = '28000';
  end if;

  select lower(u.email) into user_email from auth.users u where u.id = uid;

  select * into invite from public.organization_invitations i
  where i.token = invite_token
  for update;

  if not found or invite.accepted_at is not null or invite.expires_at < now() then
    raise exception 'Invitation not found or expired' using errcode = 'P0002';
  end if;
  if invite.email <> user_email then
    raise exception 'Invitation was sent to a different email address' using errcode = '42501';
  end if;

  insert into public.organization_members (organization_id, user_id, role)
  values (invite.organization_id, uid, invite.role)
  on conflict (organization_id, user_id) do nothing;

  update public.organization_invitations set accepted_at = now() where id = invite.id;

  update public.profiles set last_organization_id = invite.organization_id
  where id = uid and last_organization_id is null;

  return invite.organization_id;
end;
$$;

revoke execute on function public.accept_invitation(text) from public, anon;
grant execute on function public.accept_invitation(text) to authenticated;
