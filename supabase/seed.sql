-- Local development seed (`npx supabase db reset`). Users cannot be seeded
-- usefully here because sign-in goes through Supabase Auth: invite yourself
-- from Studio (Authentication → Users → Invite) and then run the bootstrap
-- snippet in docs/SUPABASE_SETUP.md.
insert into public.organizations (name, slug)
values ('Organización de desarrollo', 'dev')
on conflict (slug) do nothing;
