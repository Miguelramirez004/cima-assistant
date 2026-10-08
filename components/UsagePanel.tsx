import { Card } from "@/components/ui";
import type { UsageSummary } from "@/lib/types";

const KIND_LABELS = { formulacion: "Formulaciones", consulta: "Consultas", prospecto: "Prospectos" } as const;

/** Consumo del mes; los miembros solo ven el suyo (RLS), owners y admins el de toda la organización. */
export function UsagePanel({ usage, orgWide }: { usage: UsageSummary; orgWide: boolean }) {
  const percent = usage.quota ? Math.min(100, Math.round((usage.requestsThisMonth / usage.quota) * 100)) : 100;
  const month = new Intl.DateTimeFormat("es-ES", { month: "long", year: "numeric", timeZone: "Europe/Madrid" }).format(new Date());
  const number = new Intl.NumberFormat("es-ES");

  return (
    <Card>
      <h2 className="font-semibold">Consumo de {month}</h2>
      <p className="mb-4 text-sm text-muted">{orgWide ? "Toda la organización" : "Sus consultas este mes"}</p>

      {orgWide && (
        <div className="mb-5">
          <div className="mb-1.5 flex justify-between text-sm">
            <span><strong>{number.format(usage.requestsThisMonth)}</strong> de {number.format(usage.quota)} consultas</span>
            <span className="text-muted">{percent} %</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface ring-1 ring-line" role="progressbar"
               aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100} aria-label="Cuota mensual utilizada">
            <div className={`h-full rounded-full ${percent >= 90 ? "bg-danger" : "bg-accent"}`} style={{ width: `${percent}%` }} />
          </div>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {!orgWide && <Stat label="Total" value={number.format(usage.requestsThisMonth)} />}
        {(Object.keys(KIND_LABELS) as (keyof typeof KIND_LABELS)[]).map((kind) => (
          <Stat key={kind} label={KIND_LABELS[kind]} value={number.format(usage.byKind[kind])} />
        ))}
        {orgWide && <Stat label="Tokens" value={number.format(usage.tokensThisMonth)} />}
      </dl>

      {orgWide && usage.byMember.length > 0 && (
        <table className="mt-5 w-full text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-muted">
              <th className="py-2 font-medium">Miembro</th>
              <th className="py-2 text-right font-medium">Consultas</th>
            </tr>
          </thead>
          <tbody>
            {usage.byMember.map((m) => (
              <tr key={m.user_id} className="border-b border-line last:border-0">
                <td className="truncate py-2">{m.email ?? "Antiguo miembro"}</td>
                <td className="py-2 text-right tabular-nums">{number.format(m.requests)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-surface px-3 py-2.5 ring-1 ring-line">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-0.5 text-lg font-semibold tabular-nums">{value}</dd>
    </div>
  );
}
