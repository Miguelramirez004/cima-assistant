-- Multi-organization RLS tests (pgTAP).
-- Run with `npx supabase test db` (local Supabase) or `npm run test:db`
-- (throwaway Postgres, see scripts/test_db.sh).

begin;
create extension if not exists pgtap with schema extensions;
select plan(49);

-- ------------------------------------------------------------------- fixtures
-- Org A: alice (owner), bob (admin), carol (member). Org B: dave (owner).
-- eve has a pending invitation to A; mallory belongs nowhere.
insert into auth.users (id, email, raw_user_meta_data) values
  ('00000000-0000-0000-0000-00000000000a', 'alice@a.test', '{"full_name": "Alice"}'),
  ('00000000-0000-0000-0000-00000000000b', 'bob@a.test', '{}'),
  ('00000000-0000-0000-0000-00000000000c', 'carol@a.test', '{}'),
  ('00000000-0000-0000-0000-00000000000d', 'dave@b.test', '{}'),
  ('00000000-0000-0000-0000-00000000000e', 'eve@x.test', '{}'),
  ('00000000-0000-0000-0000-00000000000f', 'mallory@x.test', '{}');

insert into public.organizations (id, name, slug) values
  ('aaaaaaaa-0000-0000-0000-000000000000', 'Farmacia A', 'farmacia-a'),
  ('bbbbbbbb-0000-0000-0000-000000000000', 'Hospital B', 'hospital-b');

insert into public.organization_members (organization_id, user_id, role) values
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000a', 'owner'),
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000b', 'admin'),
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000c', 'member'),
  ('bbbbbbbb-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000d', 'owner');

insert into public.formulations (organization_id, user_id, query, answer) values
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000a', 'q alice', 'a'),
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000c', 'q carol', 'a'),
  ('bbbbbbbb-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000d', 'q dave', 'a');

insert into public.conversations (id, organization_id, user_id, title) values
  ('cccccccc-0000-0000-0000-000000000000', 'aaaaaaaa-0000-0000-0000-000000000000',
   '00000000-0000-0000-0000-00000000000c', 'Ibuprofeno');
insert into public.messages (conversation_id, organization_id, user_id, role, content) values
  ('cccccccc-0000-0000-0000-000000000000', 'aaaaaaaa-0000-0000-0000-000000000000',
   '00000000-0000-0000-0000-00000000000c', 'user', '¿Dosis?');

insert into public.usage_events (organization_id, user_id, kind) values
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000a', 'consulta'),
  ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000c', 'formulacion');

insert into public.organization_invitations (organization_id, email, role, token, expires_at) values
  ('aaaaaaaa-0000-0000-0000-000000000000', 'eve@x.test', 'member', 'tok-eve', now() + interval '1 day'),
  ('aaaaaaaa-0000-0000-0000-000000000000', 'mallory@x.test', 'member', 'tok-expired', now() - interval '1 day');

insert into public.cima_cache (key, value, expires_at) values
  ('fresh', '{}', now() + interval '1 hour'),
  ('stale', '{}', now() - interval '1 hour');

-- ------------------------------------------------------------- triggers / FKs
select is((select count(*)::int from public.profiles), 6, 'a profile is created for every auth user');
select is((select display_name from public.profiles where id = '00000000-0000-0000-0000-00000000000a'),
          'Alice', 'profile display_name comes from user metadata');
select is((select email from public.profiles where id = '00000000-0000-0000-0000-00000000000a'),
          'alice@a.test', 'profile email is copied from the auth user');
update auth.users set email = 'alice@new.test' where id = '00000000-0000-0000-0000-00000000000a';
select is((select email from public.profiles where id = '00000000-0000-0000-0000-00000000000a'),
          'alice@new.test', 'profile email follows email changes');
update auth.users set email = 'alice@a.test' where id = '00000000-0000-0000-0000-00000000000a';
select throws_ok(
  $$insert into public.messages (conversation_id, organization_id, user_id, role, content)
    values ('cccccccc-0000-0000-0000-000000000000', 'bbbbbbbb-0000-0000-0000-000000000000',
            '00000000-0000-0000-0000-00000000000c', 'user', 'x')$$,
  '23503', null, 'a message cannot point to a conversation of another organization');
select is(public.org_requests_this_month('aaaaaaaa-0000-0000-0000-000000000000'), 2::bigint,
          'org_requests_this_month counts the organization''s usage');

-- ------------------------------------------------------------------- anon
set local role anon;
select throws_ok('select * from public.organizations', '42501', null, 'anon cannot read organizations');
select throws_ok('select * from public.formulations', '42501', null, 'anon cannot read content');
select throws_ok($$select public.accept_invitation('tok-eve')$$, '42501', null, 'anon cannot accept invitations');
select throws_ok($$select private.is_org_member('aaaaaaaa-0000-0000-0000-000000000000')$$, '42501', null,
                 'anon cannot call the RLS helper functions');
select throws_ok($$select public.ensure_org_has_owner()$$, '42501', null,
                 'trigger functions are not callable through the API');
reset role;

-- --------------------------------------------------------- carol (member, A)
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000c", "role": "authenticated"}';

select results_eq('select slug from public.organizations', $$values ('farmacia-a')$$,
                  'member sees only their organization');
select results_eq('select query from public.formulations', $$values ('q carol')$$,
                  'member sees only their own formulations');
select is((select count(*)::int from public.messages), 1, 'member sees their own messages');
select is((select count(*)::int from public.organization_members), 3, 'member sees the members of their org only');
select is((select count(*)::int from public.profiles), 3, 'member sees profiles of fellow members only');
select results_eq('select kind from public.usage_events', $$values ('formulacion')$$,
                  'member sees only their own usage');
select is_empty('select * from public.organization_invitations', 'member cannot see invitations');
select throws_ok(
  $$insert into public.formulations (organization_id, user_id, query, answer)
    values ('aaaaaaaa-0000-0000-0000-000000000000', '00000000-0000-0000-0000-00000000000c', 'x', 'y')$$,
  '42501', null, 'users cannot write content directly (API only)');
select throws_ok(
  $$insert into public.organization_invitations (organization_id, email, invited_by)
    values ('aaaaaaaa-0000-0000-0000-000000000000', 'x@x.test', '00000000-0000-0000-0000-00000000000c')$$,
  '42501', null, 'member cannot invite');
select is_empty(
  $$update public.organization_members set role = 'admin'
    where user_id = '00000000-0000-0000-0000-00000000000c' returning 1$$,
  'member cannot change roles');
select throws_ok(
  $$update public.organizations set monthly_request_quota = 999999$$,
  '42501', null, 'nobody but the service role can change quotas');
select throws_ok(
  $$update public.profiles set is_platform_admin = true where id = '00000000-0000-0000-0000-00000000000c'$$,
  '42501', null, 'users cannot make themselves platform admin');
select lives_ok(
  $$update public.profiles set last_organization_id = 'aaaaaaaa-0000-0000-0000-000000000000'
    where id = '00000000-0000-0000-0000-00000000000c'$$,
  'user can select one of their organizations as active');
select throws_ok(
  $$update public.profiles set last_organization_id = 'bbbbbbbb-0000-0000-0000-000000000000'
    where id = '00000000-0000-0000-0000-00000000000c'$$,
  '42501', null, 'user cannot select an organization they do not belong to');
select throws_ok('select * from public.cima_cache', '42501', null, 'users cannot read the CIMA cache');
reset role;

-- ------------------------------------------------------------ bob (admin, A)
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000b", "role": "authenticated"}';

select is((select count(*)::int from public.usage_events), 2, 'admin sees the usage of the whole org');
select is_empty('select * from public.formulations', 'admin does not see other members'' formulations');
select is((select count(*)::int from public.organization_invitations), 2, 'admin sees org invitations');
select lives_ok(
  $$insert into public.organization_invitations (organization_id, email, role, invited_by)
    values ('aaaaaaaa-0000-0000-0000-000000000000', 'new@x.test', 'member', '00000000-0000-0000-0000-00000000000b')$$,
  'admin can invite members');
select throws_ok(
  $$insert into public.organization_invitations (organization_id, email, role, invited_by)
    values ('aaaaaaaa-0000-0000-0000-000000000000', 'boss@x.test', 'owner', '00000000-0000-0000-0000-00000000000b')$$,
  '42501', null, 'admin cannot invite owners');
select throws_ok(
  $$insert into public.organization_invitations (organization_id, email, role, invited_by)
    values ('bbbbbbbb-0000-0000-0000-000000000000', 'spy@x.test', 'member', '00000000-0000-0000-0000-00000000000b')$$,
  '42501', null, 'admin cannot invite into another organization');
select results_eq(
  $$update public.organization_members set role = 'admin'
    where user_id = '00000000-0000-0000-0000-00000000000c' returning role::text$$,
  $$values ('admin')$$, 'admin can promote a member to admin');
select is_empty(
  $$update public.organization_members set role = 'member'
    where user_id = '00000000-0000-0000-0000-00000000000a' returning 1$$,
  'admin cannot demote the owner');
select throws_ok(
  $$update public.organization_members set role = 'owner'
    where user_id = '00000000-0000-0000-0000-00000000000b'$$,
  '42501', null, 'admin cannot make themselves owner');
reset role;

-- ------------------------------------------------------------ dave (owner, B)
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000d", "role": "authenticated"}';

select results_eq('select slug from public.organizations', $$values ('hospital-b')$$,
                  'other org owner sees only their own organization');
select is_empty($$select * from public.organization_members
                  where organization_id = 'aaaaaaaa-0000-0000-0000-000000000000'$$,
                'other org owner cannot list org A members');
select is_empty($$update public.organizations set name = 'pwned'
                  where id = 'aaaaaaaa-0000-0000-0000-000000000000' returning 1$$,
                'other org owner cannot rename org A');
select is_empty($$delete from public.organization_members
                  where organization_id = 'aaaaaaaa-0000-0000-0000-000000000000' returning 1$$,
                'other org owner cannot remove org A members');
reset role;

-- ----------------------------------------------------------- alice (owner, A)
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000a", "role": "authenticated"}';

select throws_ok(
  $$delete from public.organization_members where user_id = '00000000-0000-0000-0000-00000000000a'$$,
  '23514', null, 'the last owner cannot leave the organization');
select lives_ok(
  $$update public.organization_members set role = 'owner' where user_id = '00000000-0000-0000-0000-00000000000b'$$,
  'owner can make another member owner');
select lives_ok(
  $$delete from public.organization_members where user_id = '00000000-0000-0000-0000-00000000000a'$$,
  'an owner can leave once another owner exists');
select is_empty('select * from public.formulations', 'after leaving, the user no longer sees their old content');
reset role;

-- --------------------------------------------------------------- invitations
set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000f", "role": "authenticated"}';
select throws_ok($$select public.accept_invitation('tok-eve')$$, '42501', null,
                 'an invitation cannot be accepted by a different email');
select throws_ok($$select public.accept_invitation('tok-expired')$$, 'P0002', null,
                 'expired invitations cannot be accepted');
reset role;

set local role authenticated;
set local request.jwt.claims = '{"sub": "00000000-0000-0000-0000-00000000000e", "role": "authenticated"}';
select is(public.accept_invitation('tok-eve'), 'aaaaaaaa-0000-0000-0000-000000000000'::uuid,
          'the invited user can accept and joins the organization');
select results_eq('select slug from public.organizations', $$values ('farmacia-a')$$,
                  'after accepting, the organization is visible');
select throws_ok($$select public.accept_invitation('tok-eve')$$, 'P0002', null,
                 'an invitation can only be used once');
reset role;

-- -------------------------------------------------------------------- cache
select is(public.purge_expired_cima_cache(), 1, 'purge removes only expired cache entries');

select * from finish();
rollback;
