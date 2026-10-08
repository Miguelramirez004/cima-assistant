-- Keep internal functions out of the public REST API (Supabase advisors
-- 0028/0029): RLS helpers move to a non-exposed `private` schema, and trigger
-- functions lose EXECUTE (triggers still fire; the privilege is only checked
-- when the trigger is created). Policies keep working: they reference the
-- helpers by OID, not by name.

create schema if not exists private;
revoke all on schema private from public;
grant usage on schema private to authenticated;

alter function public.is_org_member(uuid) set schema private;
alter function public.has_org_role(uuid, public.org_role[]) set schema private;
alter function public.is_platform_admin() set schema private;

revoke execute on function
  private.is_org_member(uuid),
  private.has_org_role(uuid, public.org_role[]),
  private.is_platform_admin()
from public, anon;
grant execute on function
  private.is_org_member(uuid),
  private.has_org_role(uuid, public.org_role[]),
  private.is_platform_admin()
to authenticated;

revoke execute on function
  public.handle_new_user(),
  public.ensure_org_has_owner(),
  public.touch_conversation()
from public, anon, authenticated;
