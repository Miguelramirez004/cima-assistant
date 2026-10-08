-- =============================================================================
-- Row Level Security and table privileges.
--
-- Principles
-- * Login required: the anon role gets no table access at all.
-- * Users see only organizations they belong to, and only their OWN content
--   inside them (org admins see usage aggregates, not other members' queries).
-- * Content is written by the API with the service role (bypasses RLS); the
--   API verifies membership of the X-Org-Id it receives on every request.
-- * Memberships are created only by accept_invitation() or the service role.
-- * Owners manage everything in their org; admins manage members and
--   invitations except anything involving the owner role.
-- =============================================================================

-- ------------------------------------------------------------------ privileges
-- Supabase grants ALL on new public tables to anon/authenticated by default;
-- replace that with explicit, minimal grants.

revoke all on
  public.organizations, public.profiles, public.organization_members,
  public.organization_invitations, public.conversations, public.messages,
  public.formulations, public.prospectos, public.usage_events, public.cima_cache
from anon, authenticated;

revoke usage, select on sequence public.usage_events_id_seq from anon, authenticated;

grant select on public.organizations to authenticated;
grant update (name) on public.organizations to authenticated;

grant select on public.profiles to authenticated;
grant update (display_name, last_organization_id) on public.profiles to authenticated;

grant select, delete on public.organization_members to authenticated;
grant update (role) on public.organization_members to authenticated;

grant select, insert, delete on public.organization_invitations to authenticated;

grant select, delete on public.conversations to authenticated;
grant update (title) on public.conversations to authenticated;
grant select on public.messages to authenticated;
grant select, delete on public.formulations to authenticated;
grant select, delete on public.prospectos to authenticated;
grant select on public.usage_events to authenticated;
-- cima_cache: no grants (service role only)

-- ------------------------------------------------------------------ enable RLS
alter table public.organizations            enable row level security;
alter table public.profiles                 enable row level security;
alter table public.organization_members     enable row level security;
alter table public.organization_invitations enable row level security;
alter table public.conversations            enable row level security;
alter table public.messages                 enable row level security;
alter table public.formulations             enable row level security;
alter table public.prospectos               enable row level security;
alter table public.usage_events             enable row level security;
alter table public.cima_cache               enable row level security;

-- ------------------------------------------------------------- organizations
create policy "organizations: members and platform admins read"
  on public.organizations for select to authenticated
  using (public.is_org_member(id) or public.is_platform_admin());

create policy "organizations: owners and admins rename"
  on public.organizations for update to authenticated
  using (public.has_org_role(id, '{owner,admin}'))
  with check (public.has_org_role(id, '{owner,admin}'));

-- ------------------------------------------------------------------ profiles
create policy "profiles: read own and fellow members"
  on public.profiles for select to authenticated
  using (
    id = (select auth.uid())
    or exists (
      select 1 from public.organization_members m
      where m.user_id = profiles.id and public.is_org_member(m.organization_id)
    )
  );

create policy "profiles: update own"
  on public.profiles for update to authenticated
  using (id = (select auth.uid()))
  with check (
    id = (select auth.uid())
    and (last_organization_id is null or public.is_org_member(last_organization_id))
  );

-- ------------------------------------------------------- organization_members
create policy "members: read memberships of own organizations"
  on public.organization_members for select to authenticated
  using (public.is_org_member(organization_id));

create policy "members: owners change any role, admins non-owner roles"
  on public.organization_members for update to authenticated
  using (
    public.has_org_role(organization_id, '{owner}')
    or (public.has_org_role(organization_id, '{admin}') and role <> 'owner')
  )
  with check (
    public.has_org_role(organization_id, '{owner}')
    or (public.has_org_role(organization_id, '{admin}') and role <> 'owner')
  );

create policy "members: leave, or remove as owner/admin"
  on public.organization_members for delete to authenticated
  using (
    user_id = (select auth.uid())
    or public.has_org_role(organization_id, '{owner}')
    or (public.has_org_role(organization_id, '{admin}') and role <> 'owner')
  );

-- --------------------------------------------------- organization_invitations
create policy "invitations: owners and admins read"
  on public.organization_invitations for select to authenticated
  using (public.has_org_role(organization_id, '{owner,admin}'));

create policy "invitations: owners invite any role, admins non-owner roles"
  on public.organization_invitations for insert to authenticated
  with check (
    invited_by = (select auth.uid())
    and accepted_at is null
    and (
      public.has_org_role(organization_id, '{owner}')
      or (public.has_org_role(organization_id, '{admin}') and role <> 'owner')
    )
  );

create policy "invitations: owners and admins revoke"
  on public.organization_invitations for delete to authenticated
  using (
    public.has_org_role(organization_id, '{owner}')
    or (public.has_org_role(organization_id, '{admin}') and role <> 'owner')
  );

-- ------------------------------------------------------------------ content
-- Own rows only, and only while still a member of that organization.

create policy "conversations: read own"
  on public.conversations for select to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));
create policy "conversations: rename own"
  on public.conversations for update to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id))
  with check (user_id = (select auth.uid()) and public.is_org_member(organization_id));
create policy "conversations: delete own"
  on public.conversations for delete to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));

create policy "messages: read own"
  on public.messages for select to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));

create policy "formulations: read own"
  on public.formulations for select to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));
create policy "formulations: delete own"
  on public.formulations for delete to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));

create policy "prospectos: read own"
  on public.prospectos for select to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));
create policy "prospectos: delete own"
  on public.prospectos for delete to authenticated
  using (user_id = (select auth.uid()) and public.is_org_member(organization_id));

-- Usage: own events, or every event of the org for owners/admins
create policy "usage_events: read own or as org admin"
  on public.usage_events for select to authenticated
  using (
    (user_id = (select auth.uid()) and public.is_org_member(organization_id))
    or public.has_org_role(organization_id, '{owner,admin}')
  );
