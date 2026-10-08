// Datos de ejemplo del modo vista previa (NEXT_PUBLIC_PREVIEW_MODE=1).

import type {
  ChatMessage, Conversation, Formulation, Invitation, Member, Membership, Organization,
  Prospecto, SessionUser, UsageSummary,
} from "@/lib/types";

const daysAgo = (days: number) => new Date(Date.now() - days * 86_400_000).toISOString();

export const PREVIEW_ORGS: Organization[] = [
  { id: "11111111-1111-4111-8111-111111111111", name: "Farmacia Central", slug: "farmacia-central",
    monthly_request_quota: 2000, created_at: daysAgo(40) },
  { id: "22222222-2222-4222-8222-222222222222", name: "Hospital Universitario", slug: "hospital-universitario",
    monthly_request_quota: 5000, created_at: daysAgo(12) },
];

export const PREVIEW_USER: SessionUser = {
  id: "00000000-0000-4000-8000-000000000001",
  email: "farmaceutica@ejemplo.es",
  displayName: "Laura Martín",
  isPlatformAdmin: true,
};

export const PREVIEW_MEMBERSHIPS: Membership[] = [
  { role: "owner", organization: PREVIEW_ORGS[0] },
  { role: "member", organization: PREVIEW_ORGS[1] },
];

const IBU_REF = { title: "IBUPROFENO KERN PHARMA 600 mg", nregistro: "62816",
  url: "https://cima.aemps.es/cima/dochtml/ft/62816/FichaTecnica.html" };
const OME_REF = { title: "OMEPRAZOL CINFA 20 mg CÁPSULAS", nregistro: "65278",
  url: "https://cima.aemps.es/cima/dochtml/ft/65278/FichaTecnica.html" };

export const PREVIEW_FORMULACION_ANSWER = `## 1. Resumen
Suspensión oral de **omeprazol 2 mg/ml** para pacientes pediátricos o con dificultad de deglución.

## 2. Composición
| Componente | Cantidad |
|---|---|
| Omeprazol (de cápsulas 20 mg) | 200 mg |
| Bicarbonato sódico 8,4 % | c.s.p. 100 ml |

## 3. Procedimiento de elaboración
1. Abrir las cápsulas y triturar los microgránulos en mortero.
2. Incorporar progresivamente el bicarbonato sódico hasta formar una pasta homogénea.
3. Completar volumen y agitar 20 minutos.

## 4. Conservación
Nevera (2–8 °C), envase topacio. **Caducidad: 30 días.**

[Ref 1: OMEPRAZOL CINFA 20 mg CÁPSULAS (Nº Registro: 65278)]`;

export const PREVIEW_PROSPECTO = `PROSPECTO: INFORMACIÓN PARA EL USUARIO

Ibuprofeno 600 mg comprimidos recubiertos con película

Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento.

1. Qué es Ibuprofeno y para qué se utiliza
Ibuprofeno pertenece al grupo de medicamentos llamados antiinflamatorios no esteroideos (AINE).

2. Qué necesita saber antes de empezar a tomar Ibuprofeno
No tome Ibuprofeno si es alérgico al ibuprofeno o tiene una úlcera péptica activa.

3. Cómo tomar Ibuprofeno
Adultos: 1 comprimido cada 8 horas. No supere 2.400 mg al día.`;

export const PREVIEW_FORMULATIONS: Formulation[] = [
  { id: "f1", query: "Suspensión oral de omeprazol 2 mg/ml para uso pediátrico",
    answer: PREVIEW_FORMULACION_ANSWER, context: "[Ref 1: OMEPRAZOL CINFA 20 mg ...]",
    references: [OME_REF], created_at: daysAgo(1) },
  { id: "f2", query: "Gel de metronidazol 0,75 % para uso tópico",
    answer: "## 1. Resumen\nGel tópico de metronidazol al 0,75 %...", context: null,
    references: [], created_at: daysAgo(4) },
];

export const PREVIEW_PROSPECTOS: Prospecto[] = [
  { id: "p1", query: "Prospecto de ibuprofeno 600 mg", medication_name: "IBUPROFENO KERN PHARMA 600 mg",
    nregistro: "62816", content: PREVIEW_PROSPECTO, created_at: daysAgo(2) },
];

export const PREVIEW_CONVERSATIONS: Conversation[] = [
  { id: "c1000000-0000-4000-8000-000000000001", title: "¿Contraindicaciones del ibuprofeno?", updated_at: daysAgo(0.2) },
  { id: "c1000000-0000-4000-8000-000000000002", title: "Interacciones de omeprazol con clopidogrel", updated_at: daysAgo(3) },
];

export const PREVIEW_MESSAGES: Record<string, ChatMessage[]> = {
  [PREVIEW_CONVERSATIONS[0].id]: [
    { id: "m1", role: "user", content: "¿Cuáles son las contraindicaciones del ibuprofeno?",
      reasoning: null, references: [], created_at: daysAgo(0.21) },
    { id: "m2", role: "assistant",
      content: "Según la ficha técnica (sección 4.3), el ibuprofeno está contraindicado en:\n\n- Hipersensibilidad al ibuprofeno o a otros AINE.\n- Úlcera péptica activa o antecedentes de hemorragia gastrointestinal.\n- Insuficiencia cardiaca grave.\n- Tercer trimestre del embarazo.\n\n[Ref 1: IBUPROFENO KERN PHARMA 600 mg (Nº Registro: 62816)]",
      reasoning: "• Intención detectada: contraindicaciones (sección 4.3 de la ficha técnica)\n• Principio activo resuelto en el catálogo oficial: IBUPROFENO (id 1066)\n• Recuperados 25 medicamentos por id exacto de principio activo; usando los 3 más relevantes",
      references: [IBU_REF], created_at: daysAgo(0.2) },
  ],
  [PREVIEW_CONVERSATIONS[1].id]: [],
};

export const PREVIEW_MEMBERS: Member[] = [
  { user_id: PREVIEW_USER.id, role: "owner", email: PREVIEW_USER.email, display_name: "Laura Martín", created_at: daysAgo(40) },
  { user_id: "u2", role: "admin", email: "jorge.ruiz@ejemplo.es", display_name: "Jorge Ruiz", created_at: daysAgo(30) },
  { user_id: "u3", role: "member", email: "ana.gomez@ejemplo.es", display_name: null, created_at: daysAgo(8) },
];

export const PREVIEW_INVITATIONS: Invitation[] = [
  { id: "i1", email: "nuevo.tecnico@ejemplo.es", role: "member", expires_at: daysAgo(-5), created_at: daysAgo(2) },
];

export const PREVIEW_USAGE: UsageSummary = {
  quota: 2000,
  requestsThisMonth: 412,
  tokensThisMonth: 1_284_550,
  byKind: { formulacion: 156, consulta: 221, prospecto: 35 },
  byMember: [
    { user_id: PREVIEW_USER.id, email: PREVIEW_USER.email, requests: 230 },
    { user_id: "u2", email: "jorge.ruiz@ejemplo.es", requests: 141 },
    { user_id: "u3", email: "ana.gomez@ejemplo.es", requests: 41 },
  ],
};

export const PREVIEW_ORG_COOKIE = "preview_org";
