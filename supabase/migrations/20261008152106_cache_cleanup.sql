-- Nightly purge of expired CIMA cache entries with pg_cron, when available
-- (it is on Supabase; plain local Postgres used in CI may not have it).

create function public.purge_expired_cima_cache()
returns integer
language sql security definer set search_path = ''
as $$
  with deleted as (
    delete from public.cima_cache where expires_at < now() returning 1
  )
  select count(*)::integer from deleted;
$$;

revoke execute on function public.purge_expired_cima_cache() from public, anon, authenticated;
grant execute on function public.purge_expired_cima_cache() to service_role;

do $$
begin
  if exists (select 1 from pg_available_extensions where name = 'pg_cron') then
    create extension if not exists pg_cron with schema pg_catalog;
    perform cron.schedule(
      'purge-expired-cima-cache',
      '17 3 * * *',
      'select public.purge_expired_cima_cache()'
    );
  else
    raise notice 'pg_cron not available: schedule public.purge_expired_cima_cache() elsewhere';
  end if;
end;
$$;
