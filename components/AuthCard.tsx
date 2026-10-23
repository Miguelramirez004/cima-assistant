import type { ReactNode } from "react";

import { Logo } from "@/components/Sidebar";

/** Marco de las páginas sin sesión (login, invitación). */
export function AuthCard({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children: ReactNode }) {
  return (
    <main className="relative isolate flex min-h-screen items-center justify-center bg-background px-4 py-12">
      <div aria-hidden="true" className="cima-reticula absolute inset-x-0 top-0 -z-10 h-[520px]" />
      <div className="cima-entra w-full max-w-md">
        <div className="mb-6 flex justify-center"><Logo /></div>
        <div className="rounded-2xl border border-line bg-white p-7 shadow-[0_12px_28px_-20px_rgba(20,60,48,0.35)]">
          <h1 className="text-xl font-bold tracking-tight">{title}</h1>
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
