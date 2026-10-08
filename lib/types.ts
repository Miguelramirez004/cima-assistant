// Tipos del dominio compartidos por páginas y componentes.

export type OrgRole = "owner" | "admin" | "member";

export interface Organization {
  id: string;
  name: string;
  slug: string;
  monthly_request_quota: number;
  created_at?: string;
}

export interface Membership {
  role: OrgRole;
  organization: Organization;
}

export interface SessionUser {
  id: string;
  email: string;
  displayName: string | null;
  isPlatformAdmin: boolean;
}

export interface SessionContext {
  user: SessionUser;
  memberships: Membership[];
  activeOrg: Organization | null;
  activeRole: OrgRole | null;
}

export interface Reference {
  title: string;
  url: string;
  nregistro?: string | null;
}

export interface Formulation {
  id: string;
  query: string;
  answer: string;
  context: string | null;
  references: Reference[];
  created_at: string;
}

export interface Prospecto {
  id: string;
  query: string;
  medication_name: string | null;
  nregistro: string | null;
  content: string;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string | null;
  references: Reference[];
  created_at: string;
}

export interface Member {
  user_id: string;
  role: OrgRole;
  email: string | null;
  display_name: string | null;
  created_at: string;
}

export interface Invitation {
  id: string;
  email: string;
  role: OrgRole;
  expires_at: string;
  created_at: string;
}

export interface UsageSummary {
  quota: number;
  requestsThisMonth: number;
  tokensThisMonth: number;
  byKind: Record<"formulacion" | "consulta" | "prospecto", number>;
  byMember: { user_id: string; email: string | null; requests: number }[];
}

export interface FormulacionResult {
  id: string | null;
  answer: string;
  context: string;
  references: Reference[];
  redirect: "prospecto" | null;
  success: boolean;
}

export interface ProspectoResult {
  id: string | null;
  content: string;
  context: string;
  medication_name: string | null;
  nregistro: string | null;
  success: boolean;
}

export type ConsultaEvent =
  | { type: "conversation"; conversation_id: string; user_message_id: string }
  | { type: "trace"; message: string }
  | { type: "references"; references: Reference[] }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; reasoning: string; references: Reference[]; success: boolean; message_id: string }
  | { type: "error"; detail: string };

export const ROLE_LABELS: Record<OrgRole, string> = {
  owner: "Propietario",
  admin: "Administrador",
  member: "Miembro",
};
