import type { ReactNode } from "react";

/** Marco de las páginas sin sesión (login, invitación). */
export function AuthCard({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children: ReactNode }) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-surface px-4 py-12">
      <div className="w-full max-w-md">
        <p className="mb-6 text-center text-xs font-semibold uppercase tracking-wider text-accent">CIMA Assistant</p>
        <div className="rounded-2xl border border-line bg-white p-7 shadow-sm">
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-1.5 text-sm text-muted">{subtitle}</p>}
          <div className="mt-6">{children}</div>
        </div>
        <p className="mt-6 text-center text-xs text-muted">
          Información oficial de CIMA (AEMPS). No sustituye el criterio de un profesional sanitario.
        </p>
      </div>
    </main>
  );
}
