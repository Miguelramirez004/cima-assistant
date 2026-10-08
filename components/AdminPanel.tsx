"use client";

import { useState } from "react";

import { Alert, Button, Card, formatDate, inputClass } from "@/components/ui";
import { type InvitationResult, createOrganization } from "@/lib/client-actions";
import type { Organization } from "@/lib/types";

function toSlug(name: string) {
  return name.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
}

export function AdminPanel({ organizations: initial }: { organizations: Organization[] }) {
  const [organizations, setOrganizations] = useState(initial);
  const [name, setName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugEdited, setSlugEdited] = useState(false);
  const [ownerEmail, setOwnerEmail] = useState("");
  const [quota, setQuota] = useState(2000);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<InvitationResult | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    setCreated(null);
    try {
      const result = await createOrganization({ name: name.trim(), slug, owner_email: ownerEmail.trim(), monthly_request_quota: quota });
      setOrganizations((list) => [{ id: result.organization_id, name: name.trim(), slug, monthly_request_quota: quota,
                                    created_at: new Date().toISOString() }, ...list]);
      setCreated(result.invitation);
      setName(""); setSlug(""); setSlugEdited(false); setOwnerEmail(""); setQuota(2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo crear la organización");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <h2 className="mb-4 font-semibold">Nueva organización</h2>
        <form onSubmit={submit} className="grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="org-name" className="mb-1.5 block text-sm font-medium">Nombre</label>
            <input id="org-name" required minLength={2} maxLength={120} value={name} className={inputClass}
                   placeholder="Farmacia Central"
                   onChange={(e) => { setName(e.target.value); if (!slugEdited) setSlug(toSlug(e.target.value)); }} />
          </div>
          <div>
            <label htmlFor="org-slug" className="mb-1.5 block text-sm font-medium">Identificador</label>
            <input id="org-slug" required pattern="[a-z0-9]+(-[a-z0-9]+)*" maxLength={60} value={slug}
                   className={`${inputClass} font-mono`} placeholder="farmacia-central"
                   onChange={(e) => { setSlug(e.target.value); setSlugEdited(true); }} />
          </div>
          <div>
            <label htmlFor="org-owner" className="mb-1.5 block text-sm font-medium">Email del propietario</label>
            <input id="org-owner" type="email" required value={ownerEmail} className={inputClass}
                   placeholder="titular@farmacia.es" onChange={(e) => setOwnerEmail(e.target.value)} />
          </div>
          <div>
            <label htmlFor="org-quota" className="mb-1.5 block text-sm font-medium">Cuota mensual (consultas)</label>
            <input id="org-quota" type="number" min={0} step={100} required value={quota} className={inputClass}
                   onChange={(e) => setQuota(Number(e.target.value))} />
          </div>
          <div className="sm:col-span-2">
            <Button type="submit" disabled={saving}>{saving ? "Creando…" : "Crear e invitar al propietario"}</Button>
          </div>
        </form>
        <div className="mt-4 space-y-3">
          {error && <Alert>{error}</Alert>}
          {created && (
            <Alert tone={created.email_sent ? "success" : "info"}>
              Organización creada. {created.email_sent
                ? <>Invitación enviada a <strong>{created.email}</strong>.</>
                : <>No se pudo enviar el email; comparta este enlace con <strong>{created.email}</strong>:</>}
              <input readOnly value={created.invite_url} onFocus={(e) => e.target.select()}
                     className="mt-2 w-full rounded border border-line bg-white px-2 py-1 font-mono text-xs text-foreground" />
            </Alert>
          )}
        </div>
      </Card>

      <Card>
        <h2 className="mb-4 font-semibold">Organizaciones <span className="font-normal text-muted">({organizations.length})</span></h2>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[480px] text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-muted">
                <th className="py-2 font-medium">Nombre</th>
                <th className="py-2 font-medium">Identificador</th>
                <th className="py-2 text-right font-medium">Cuota</th>
                <th className="py-2 text-right font-medium">Alta</th>
              </tr>
            </thead>
            <tbody>
              {organizations.map((o) => (
                <tr key={o.id} className="border-b border-line last:border-0">
                  <td className="py-2 font-medium">{o.name}</td>
                  <td className="py-2 font-mono text-xs text-muted">{o.slug}</td>
                  <td className="py-2 text-right tabular-nums">{new Intl.NumberFormat("es-ES").format(o.monthly_request_quota)}</td>
                  <td className="py-2 text-right text-xs text-muted">{o.created_at ? formatDate(o.created_at) : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted">Las cuotas se cambian en Supabase (tabla organizations).</p>
      </Card>
    </div>
  );
}
