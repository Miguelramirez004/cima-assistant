"use client";

// Acciones del navegador: llamadas a la API (/api/*) con el token de Supabase
// y la organización activa, y operaciones de cuenta. En modo vista previa se
// simulan con datos de ejemplo.

import { IS_PREVIEW } from "@/lib/env";
import * as fx from "@/lib/preview/fixtures";
import { getSupabaseBrowserClient } from "@/lib/supabase/client";
import type {
  ConsultaEvent, FormulacionResult, Invitation, OrgRole, ProspectoResult,
} from "@/lib/types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function authHeaders(orgId?: string): Promise<Record<string, string>> {
  const { data } = await getSupabaseBrowserClient().auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new ApiError(401, "La sesión ha caducado. Vuelva a iniciar sesión.");
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
    ...(orgId ? { "X-Org-Id": orgId } : {}),
  };
}

async function errorFrom(response: Response): Promise<ApiError> {
  let detail = `Error ${response.status}`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join(". ");
  } catch {
    // respuesta sin JSON
  }
  return new ApiError(response.status, detail);
}

async function postJson<T>(path: string, body: unknown, orgId?: string): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: await authHeaders(orgId),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

// ------------------------------------------------------------- generación

export async function generateFormulacion(orgId: string, query: string, advancedSearch: boolean): Promise<FormulacionResult> {
  if (IS_PREVIEW) {
    await sleep(1200);
    if (/prospecto/i.test(query)) {
      return { id: null, answer: "", context: "", references: [], redirect: "prospecto", success: true };
    }
    return { id: "preview", answer: fx.PREVIEW_FORMULACION_ANSWER, context: "[Ref 1: OMEPRAZOL CINFA 20 mg ...]",
             references: fx.PREVIEW_FORMULATIONS[0].references, redirect: null, success: true };
  }
  return postJson("/api/formulacion", { query, advanced_search: advancedSearch }, orgId);
}

export async function generateProspecto(orgId: string, query: string): Promise<ProspectoResult> {
  if (IS_PREVIEW) {
    await sleep(1200);
    return { id: "preview", content: fx.PREVIEW_PROSPECTO, context: "", success: true,
             medication_name: "IBUPROFENO KERN PHARMA 600 mg", nregistro: "62816" };
  }
  return postJson("/api/prospecto", { query }, orgId);
}

/** Consulta en streaming: llama a `onEvent` por cada evento SSE recibido. */
export async function streamConsulta(
  orgId: string, question: string, conversationId: string | null,
  onEvent: (event: ConsultaEvent) => void, signal?: AbortSignal,
): Promise<void> {
  if (IS_PREVIEW) return previewStream(question, conversationId, onEvent);

  const response = await fetch("/api/consulta", {
    method: "POST",
    headers: await authHeaders(orgId),
    body: JSON.stringify({ question, conversation_id: conversationId }),
    signal,
  });
  if (!response.ok || !response.body) throw await errorFrom(response);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseSseBlock(block);
      if (event) onEvent(event);
    }
  }
}

export function parseSseBlock(block: string): ConsultaEvent | null {
  let type = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { type, ...JSON.parse(data.join("\n")) } as ConsultaEvent;
  } catch {
    return null;
  }
}

async function previewStream(question: string, conversationId: string | null, onEvent: (e: ConsultaEvent) => void) {
  const answer = fx.PREVIEW_MESSAGES[fx.PREVIEW_CONVERSATIONS[0].id][1];
  onEvent({ type: "conversation", conversation_id: conversationId ?? crypto.randomUUID(), user_message_id: "u" });
  for (const step of (answer.reasoning ?? "").split("\n")) {
    await sleep(450);
    onEvent({ type: "trace", message: step.replace(/^• /, "") });
  }
  onEvent({ type: "references", references: answer.references });
  for (const word of answer.content.split(/(\s+)/)) {
    await sleep(25);
    onEvent({ type: "token", text: word });
  }
  onEvent({ type: "done", answer: answer.content, reasoning: answer.reasoning ?? "",
            references: answer.references, success: true, message_id: crypto.randomUUID() });
  void question;
}

// --------------------------------------------------------- organizaciones

export interface InvitationResult extends Pick<Invitation, "id" | "email" | "role" | "expires_at"> {
  invite_url: string;
  email_sent: boolean;
}

export async function inviteMember(orgId: string, email: string, role: OrgRole): Promise<InvitationResult> {
  if (IS_PREVIEW) {
    await sleep(600);
    return { id: "preview", email, role, expires_at: new Date(Date.now() + 7 * 86_400_000).toISOString(),
             invite_url: `${location.origin}/invite/preview-token`, email_sent: true };
  }
  return postJson(`/api/organizations/${orgId}/invitations`, { email, role });
}

export async function createOrganization(input: {
  name: string; slug: string; owner_email: string; monthly_request_quota: number;
}): Promise<{ organization_id: string; invitation: InvitationResult }> {
  if (IS_PREVIEW) {
    await sleep(600);
    return { organization_id: "preview", invitation: await inviteMember("preview", input.owner_email, "owner") };
  }
  return postJson("/api/admin/organizations", input);
}

/** Operaciones directas sobre tablas: RLS decide si el usuario puede hacerlas. */
async function mutate(run: () => PromiseLike<{ error: { message: string } | null }>) {
  if (IS_PREVIEW) {
    await sleep(300);
    return;
  }
  const { error } = await run();
  if (error) {
    throw new ApiError(400, /at least one owner/i.test(error.message)
      ? "La organización debe tener al menos un propietario. Nombre a otro propietario antes."
      : error.message);
  }
}

export const changeMemberRole = (orgId: string, userId: string, role: OrgRole) =>
  mutate(() => getSupabaseBrowserClient().from("organization_members")
    .update({ role }).eq("organization_id", orgId).eq("user_id", userId));

export const removeMember = (orgId: string, userId: string) =>
  mutate(() => getSupabaseBrowserClient().from("organization_members")
    .delete().eq("organization_id", orgId).eq("user_id", userId));

export const revokeInvitation = (invitationId: string) =>
  mutate(() => getSupabaseBrowserClient().from("organization_invitations").delete().eq("id", invitationId));

export const renameOrganization = (orgId: string, name: string) =>
  mutate(() => getSupabaseBrowserClient().from("organizations").update({ name }).eq("id", orgId));

export const deleteHistoryItem = (table: "formulations" | "prospectos" | "conversations", id: string) =>
  mutate(() => getSupabaseBrowserClient().from(table).delete().eq("id", id));

// ------------------------------------------------------------------ cuenta

export async function switchOrganization(userId: string, orgId: string): Promise<void> {
  if (IS_PREVIEW) {
    document.cookie = `${fx.PREVIEW_ORG_COOKIE}=${orgId}; path=/; max-age=31536000; samesite=lax`;
    return;
  }
  await mutate(() => getSupabaseBrowserClient().from("profiles")
    .update({ last_organization_id: orgId }).eq("id", userId));
}

export async function sendLoginLink(email: string, next: string): Promise<void> {
  if (IS_PREVIEW) {
    await sleep(500);
    return;
  }
  const redirect = `${location.origin}/auth/callback?next=${encodeURIComponent(next)}`;
  const { error } = await getSupabaseBrowserClient().auth.signInWithOtp({
    email,
    options: { shouldCreateUser: false, emailRedirectTo: redirect },
  });
  if (error) {
    const notAllowed = /signups? not allowed|not found|user not found/i.test(error.message);
    throw new ApiError(error.status ?? 400, notAllowed
      ? "Este email no tiene acceso. Pida una invitación al responsable de su organización."
      : error.message);
  }
}

export async function signOut(): Promise<void> {
  if (!IS_PREVIEW) await getSupabaseBrowserClient().auth.signOut();
}

export async function acceptInvitation(token: string): Promise<string> {
  if (IS_PREVIEW) {
    await sleep(600);
    return fx.PREVIEW_ORGS[0].id;
  }
  const supabase = getSupabaseBrowserClient();
  const { data, error } = await supabase.rpc("accept_invitation", { invite_token: token });
  if (error) {
    const message = /expired|not found/i.test(error.message)
      ? "La invitación no existe, ya se usó o ha caducado."
      : /different email/i.test(error.message)
        ? "Esta invitación se envió a otro email. Inicie sesión con el email invitado."
        : error.message;
    throw new ApiError(400, message);
  }
  const orgId = data as string;
  const { data: user } = await supabase.auth.getUser();
  if (user.user) {
    await supabase.from("profiles").update({ last_organization_id: orgId }).eq("id", user.user.id);
  }
  return orgId;
}
