// Capa de datos del servidor: lecturas con el cliente de Supabase del usuario
// (RLS decide qué ve), o datos de ejemplo en modo vista previa.

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { cache } from "react";

import { IS_PREVIEW } from "@/lib/env";
import * as fx from "@/lib/preview/fixtures";
import { getSupabaseServerClient } from "@/lib/supabase/server";
import type {
  ChatMessage, Conversation, Formulation, Invitation, Member, Membership, Organization,
  Prospecto, SessionContext, UsageSummary,
} from "@/lib/types";

function pickActive(memberships: Membership[], preferredId: string | null | undefined) {
  return memberships.find((m) => m.organization.id === preferredId) ?? memberships[0] ?? null;
}

/** Usuario, organizaciones y organización activa. Redirige a /login sin sesión. */
export const getSessionContext = cache(async (): Promise<SessionContext> => {
  if (IS_PREVIEW) {
    const preferred = (await cookies()).get(fx.PREVIEW_ORG_COOKIE)?.value;
    const active = pickActive(fx.PREVIEW_MEMBERSHIPS, preferred);
    return { user: fx.PREVIEW_USER, memberships: fx.PREVIEW_MEMBERSHIPS,
             activeOrg: active?.organization ?? null, activeRole: active?.role ?? null };
  }

  const supabase = await getSupabaseServerClient();
  const { data: claimsData } = await supabase.auth.getClaims();
  const claims = claimsData?.claims;
  if (!claims?.sub) redirect("/login");

  const [{ data: profile }, { data: rows }] = await Promise.all([
    supabase.from("profiles")
      .select("display_name, email, is_platform_admin, last_organization_id")
      .eq("id", claims.sub).maybeSingle(),
    supabase.from("organization_members")
      .select("role, organization:organizations(id, name, slug, monthly_request_quota, created_at)")
      .eq("user_id", claims.sub),
  ]);

  const memberships = ((rows ?? []) as unknown as Membership[])
    .filter((m) => m.organization)
    .sort((a, b) => a.organization.name.localeCompare(b.organization.name, "es"));
  const active = pickActive(memberships, profile?.last_organization_id);

  return {
    user: {
      id: claims.sub,
      email: (claims.email as string | undefined) ?? profile?.email ?? "",
      displayName: profile?.display_name ?? null,
      isPlatformAdmin: Boolean(profile?.is_platform_admin),
    },
    memberships,
    activeOrg: active?.organization ?? null,
    activeRole: active?.role ?? null,
  };
});

export async function listFormulations(orgId: string, limit = 50): Promise<Formulation[]> {
  if (IS_PREVIEW) return fx.PREVIEW_FORMULATIONS;
  const supabase = await getSupabaseServerClient();
  const { data } = await supabase.from("formulations")
    .select("id, query, answer, context, references, created_at")
    .eq("organization_id", orgId).order("created_at", { ascending: false }).limit(limit);
  return (data ?? []) as Formulation[];
}

export async function listProspectos(orgId: string, limit = 50): Promise<Prospecto[]> {
  if (IS_PREVIEW) return fx.PREVIEW_PROSPECTOS;
  const supabase = await getSupabaseServerClient();
  const { data } = await supabase.from("prospectos")
    .select("id, query, medication_name, nregistro, content, created_at")
    .eq("organization_id", orgId).order("created_at", { ascending: false }).limit(limit);
  return (data ?? []) as Prospecto[];
}

export async function listConversations(orgId: string, limit = 50): Promise<Conversation[]> {
  if (IS_PREVIEW) return fx.PREVIEW_CONVERSATIONS;
  const supabase = await getSupabaseServerClient();
  const { data } = await supabase.from("conversations")
    .select("id, title, updated_at")
    .eq("organization_id", orgId).order("updated_at", { ascending: false }).limit(limit);
  return (data ?? []) as Conversation[];
}

/** Mensajes de una conversación del usuario en la organización (null si no existe). */
export async function getConversationMessages(orgId: string, conversationId: string): Promise<ChatMessage[] | null> {
  if (IS_PREVIEW) return fx.PREVIEW_MESSAGES[conversationId] ?? null;
  const supabase = await getSupabaseServerClient();
  const { data: conversation } = await supabase.from("conversations")
    .select("id").eq("id", conversationId).eq("organization_id", orgId).maybeSingle();
  if (!conversation) return null;
  const { data } = await supabase.from("messages")
    .select("id, role, content, reasoning, references, created_at")
    .eq("conversation_id", conversationId).order("created_at", { ascending: true });
  return (data ?? []) as ChatMessage[];
}

export async function listMembers(orgId: string): Promise<Member[]> {
  if (IS_PREVIEW) return fx.PREVIEW_MEMBERS;
  const supabase = await getSupabaseServerClient();
  const { data: rows } = await supabase.from("organization_members")
    .select("user_id, role, created_at").eq("organization_id", orgId).order("created_at");
  const members = rows ?? [];
  const { data: profiles } = members.length
    ? await supabase.from("profiles").select("id, email, display_name").in("id", members.map((m) => m.user_id))
    : { data: [] };
  const byId = new Map((profiles ?? []).map((p) => [p.id, p]));
  return members.map((m) => ({
    ...m,
    email: byId.get(m.user_id)?.email ?? null,
    display_name: byId.get(m.user_id)?.display_name ?? null,
  })) as Member[];
}

export async function listPendingInvitations(orgId: string): Promise<Invitation[]> {
  if (IS_PREVIEW) return fx.PREVIEW_INVITATIONS;
  const supabase = await getSupabaseServerClient();
  const { data } = await supabase.from("organization_invitations")
    .select("id, email, role, expires_at, created_at")
    .eq("organization_id", orgId).is("accepted_at", null).gt("expires_at", new Date().toISOString())
    .order("created_at", { ascending: false });
  return (data ?? []) as Invitation[];
}

/** Consumo del mes en curso. RLS limita a los propios eventos salvo para owners/admins. */
export async function getUsageSummary(org: Organization, members: Member[]): Promise<UsageSummary> {
  if (IS_PREVIEW) return fx.PREVIEW_USAGE;
  const supabase = await getSupabaseServerClient();
  const monthStart = new Date();
  monthStart.setUTCDate(1);
  monthStart.setUTCHours(0, 0, 0, 0);
  const { data } = await supabase.from("usage_events")
    .select("user_id, kind, prompt_tokens, completion_tokens")
    .eq("organization_id", org.id).gte("created_at", monthStart.toISOString()).limit(10_000);

  const events = data ?? [];
  const byKind = { formulacion: 0, consulta: 0, prospecto: 0 };
  const perUser = new Map<string, number>();
  let tokens = 0;
  for (const e of events) {
    byKind[e.kind as keyof typeof byKind] += 1;
    perUser.set(e.user_id, (perUser.get(e.user_id) ?? 0) + 1);
    tokens += (e.prompt_tokens ?? 0) + (e.completion_tokens ?? 0);
  }
  const emails = new Map(members.map((m) => [m.user_id, m.email]));
  return {
    quota: org.monthly_request_quota,
    requestsThisMonth: events.length,
    tokensThisMonth: tokens,
    byKind,
    byMember: [...perUser.entries()]
      .map(([user_id, requests]) => ({ user_id, email: emails.get(user_id) ?? null, requests }))
      .sort((a, b) => b.requests - a.requests),
  };
}

/** Todas las organizaciones (solo administradores de plataforma, por RLS). */
export async function listAllOrganizations(): Promise<Organization[]> {
  if (IS_PREVIEW) return fx.PREVIEW_ORGS;
  const supabase = await getSupabaseServerClient();
  const { data } = await supabase.from("organizations")
    .select("id, name, slug, monthly_request_quota, created_at").order("created_at", { ascending: false });
  return (data ?? []) as Organization[];
}
