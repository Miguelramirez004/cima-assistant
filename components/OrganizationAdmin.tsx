"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { useActiveOrg, useSession } from "@/components/SessionProvider";
import { Alert, Badge, Button, Card, formatDate, inputClass } from "@/components/ui";
import {
  type InvitationResult, changeMemberRole, inviteMember, removeMember, renameOrganization, revokeInvitation,
} from "@/lib/client-actions";
import { ROLE_LABELS, type Invitation, type Member, type OrgRole } from "@/lib/types";

export function OrganizationAdmin({ members: initialMembers, invitations: initialInvitations }: {
  members: Member[]; invitations: Invitation[];
}) {
  const { user } = useSession();
  const { org, role } = useActiveOrg();
  const router = useRouter();
  const canManage = role === "owner" || role === "admin";
  const [members, setMembers] = useState(initialMembers);
  const [invitations, setInvitations] = useState(initialInvitations);
  const [error, setError] = useState<string | null>(null);

  // Lo que el rol actual puede hacer con cada miembro (espejo de las políticas RLS)
  const canEdit = (m: Member) => canManage && m.user_id !== user.id && (role === "owner" || m.role !== "owner");
  const assignableRoles: OrgRole[] = role === "owner" ? ["owner", "admin", "member"] : ["admin", "member"];

  const run = async (action: () => Promise<void>) => {
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo completar la acción");
    }
  };

  const onRoleChange = (m: Member, newRole: OrgRole) => run(async () => {
    await changeMemberRole(org.id, m.user_id, newRole);
    setMembers((list) => list.map((x) => (x.user_id === m.user_id ? { ...x, role: newRole } : x)));
  });

  const onRemove = (m: Member) => run(async () => {
    if (!confirm(`¿Quitar a ${m.email ?? "este miembro"} de ${org.name}?`)) return;
    await removeMember(org.id, m.user_id);
    setMembers((list) => list.filter((x) => x.user_id !== m.user_id));
  });

  const onLeave = () => run(async () => {
    if (!confirm(`¿Salir de ${org.name}? Perderá el acceso a su historial en esta organización.`)) return;
    await removeMember(org.id, user.id);
    router.push("/");
    router.refresh();
  });

  return (
    <div className="space-y-6">
      {error && <Alert>{error}</Alert>}
      {canManage && <RenameForm initialName={org.name} onSave={(name) => run(() => renameOrganization(org.id, name).then(() => router.refresh()))} />}

      <Card>
        <h2 className="mb-4 font-semibold">Miembros <span className="font-normal text-muted">({members.length})</span></h2>
        <ul className="divide-y divide-line">
          {members.map((m) => (
            <li key={m.user_id} className="flex flex-wrap items-center justify-between gap-3 py-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">
                  {m.display_name ?? m.email ?? "Usuario"} {m.user_id === user.id && <span className="text-muted">(usted)</span>}
                </p>
                {m.display_name && <p className="truncate text-xs text-muted">{m.email}</p>}
              </div>
              <div className="flex items-center gap-2">
                {canEdit(m) ? (
                  <>
                    <label className="sr-only" htmlFor={`role-${m.user_id}`}>Rol</label>
                    <select id={`role-${m.user_id}`} value={m.role} onChange={(e) => onRoleChange(m, e.target.value as OrgRole)}
                            className="rounded-md border border-line bg-white px-2 py-1 text-sm">
                      {assignableRoles.map((r) => <option key={r} value={r}>{ROLE_LABELS[r]}</option>)}
                    </select>
                    <Button variant="ghost" onClick={() => onRemove(m)}>Quitar</Button>
                  </>
                ) : (
                  <Badge tone={m.role === "member" ? "neutral" : "accent"}>{ROLE_LABELS[m.role]}</Badge>
                )}
              </div>
            </li>
          ))}
        </ul>
        <div className="mt-4 border-t border-line pt-4">
          <Button variant="ghost" onClick={onLeave}>Salir de la organización</Button>
        </div>
      </Card>

      {canManage && (
        <Card>
          <h2 className="mb-4 font-semibold">Invitar a un miembro</h2>
          <InviteForm orgId={org.id} roles={assignableRoles}
                      onInvited={(inv) => setInvitations((list) => [
                        { id: inv.id, email: inv.email, role: inv.role, expires_at: inv.expires_at, created_at: new Date().toISOString() },
                        ...list,
                      ])} />
          {invitations.length > 0 && (
            <>
              <h3 className="mb-2 mt-6 text-sm font-semibold">Invitaciones pendientes</h3>
              <ul className="divide-y divide-line rounded-lg border border-line">
                {invitations.map((inv) => (
                  <li key={inv.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5">
                    <div className="min-w-0">
                      <p className="truncate text-sm">{inv.email}</p>
                      <p className="text-xs text-muted">{ROLE_LABELS[inv.role]} · caduca {formatDate(inv.expires_at)}</p>
                    </div>
                    {(role === "owner" || inv.role !== "owner") && (
                      <Button variant="ghost" onClick={() => run(async () => {
                        await revokeInvitation(inv.id);
                        setInvitations((list) => list.filter((x) => x.id !== inv.id));
                      })}>Revocar</Button>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}
        </Card>
      )}
    </div>
  );
}

function RenameForm({ initialName, onSave }: { initialName: string; onSave: (name: string) => Promise<void> }) {
  const [name, setName] = useState(initialName);
  const [saving, setSaving] = useState(false);
  return (
    <Card>
      <form className="flex flex-wrap items-end gap-3" onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        await onSave(name.trim());
        setSaving(false);
      }}>
        <div className="min-w-60 flex-1">
          <label htmlFor="org-name" className="mb-1.5 block text-sm font-medium">Nombre de la organización</label>
          <input id="org-name" value={name} minLength={2} maxLength={120} required
                 onChange={(e) => setName(e.target.value)} className={inputClass} />
        </div>
        <Button type="submit" variant="secondary" disabled={saving || name.trim() === initialName || name.trim().length < 2}>
          {saving ? "Guardando…" : "Guardar"}
        </Button>
      </form>
    </Card>
  );
}

export function InviteForm({ orgId, roles, onInvited }: {
  orgId: string; roles: OrgRole[]; onInvited?: (invitation: InvitationResult) => void;
}) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<OrgRole>("member");
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState<InvitationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  return (
    <form className="space-y-3" onSubmit={async (e) => {
      e.preventDefault();
      setSending(true);
      setError(null);
      setResult(null);
      try {
        const invitation = await inviteMember(orgId, email.trim(), role);
        setResult(invitation);
        setEmail("");
        onInvited?.(invitation);
      } catch (err) {
        setError(err instanceof Error ? err.message : "No se pudo enviar la invitación");
      } finally {
        setSending(false);
      }
    }}>
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-60 flex-1">
          <label htmlFor="invite-email" className="mb-1.5 block text-sm font-medium">Email</label>
          <input id="invite-email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                 placeholder="nombre@farmacia.es" className={inputClass} />
        </div>
        <div>
          <label htmlFor="invite-role" className="mb-1.5 block text-sm font-medium">Rol</label>
          <select id="invite-role" value={role} onChange={(e) => setRole(e.target.value as OrgRole)}
                  className={`${inputClass} w-auto`}>
            {roles.map((r) => <option key={r} value={r}>{ROLE_LABELS[r]}</option>)}
          </select>
        </div>
        <Button type="submit" disabled={sending}>{sending ? "Enviando…" : "Enviar invitación"}</Button>
      </div>
      {error && <Alert>{error}</Alert>}
      {result && (
        <Alert tone={result.email_sent ? "success" : "info"}>
          {result.email_sent
            ? <>Invitación enviada a <strong>{result.email}</strong>.</>
            : <>No se pudo enviar el email. Comparta este enlace con <strong>{result.email}</strong>:</>}
          <input readOnly value={result.invite_url} onFocus={(e) => e.target.select()}
                 className="mt-2 w-full rounded border border-line bg-white px-2 py-1 font-mono text-xs text-foreground" />
        </Alert>
      )}
    </form>
  );
}
