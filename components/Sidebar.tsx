"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { IndicadorConexion } from "@/components/motion";
import { useSession } from "@/components/SessionProvider";
import { signOut, switchOrganization } from "@/lib/client-actions";
import { IS_PREVIEW } from "@/lib/env";
import { ROLE_LABELS } from "@/lib/types";

// Iconos de línea (24×24) por sección
const ICONS: Record<string, string> = {
  "/formulacion": "M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3",
  "/consultas": "M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z",
  "/prospectos": "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  "/historial": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3 2",
  "/organizacion": "M4 21V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v16M16 9h2a2 2 0 0 1 2 2v10M8 7h4M8 11h4M8 15h4M3 21h18",
  "/admin": "M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6z",
};

export function Logo() {
  return (
    <span className="flex items-center gap-3">
      <span aria-hidden="true" className="flex size-9 items-center justify-center rounded-[10px] bg-accent">
        <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="#fff" strokeWidth={2.4} strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
      </span>
      <span className="flex flex-col">
        <span className="text-[15px] font-extrabold tracking-tight">CIMA Assistant</span>
        <span className="text-xs text-muted">Información oficial AEMPS</span>
      </span>
    </span>
  );
}

const NAV = [
  { href: "/formulacion", label: "Formulación magistral" },
  { href: "/consultas", label: "Consultas CIMA" },
  { href: "/prospectos", label: "Prospectos" },
  { href: "/historial", label: "Historial" },
];

export function Sidebar() {
  const { user, memberships, activeOrg, activeRole } = useSession();
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [switching, startSwitch] = useTransition();

  const links = [
    ...(activeOrg ? NAV : []),
    ...(activeOrg ? [{ href: "/organizacion", label: "Organización" }] : []),
    ...(user.isPlatformAdmin ? [{ href: "/admin", label: "Administración" }] : []),
  ];

  const onSwitch = (orgId: string) => startSwitch(async () => {
    await switchOrganization(user.id, orgId);
    router.refresh();
  });

  const onSignOut = async () => {
    await signOut();
    router.push("/login");
    router.refresh();
  };

  return (
    <>
      {/* Barra superior en móvil */}
      <div className="flex items-center justify-between border-b border-line bg-surface px-4 py-3 md:hidden">
        <Logo />
        <button onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="sidebar"
                className="rounded-md border border-line px-3 py-1 text-sm">
          {open ? "Cerrar" : "Menú"}
        </button>
      </div>

      <aside id="sidebar"
             className={`${open ? "flex" : "hidden"} w-full flex-col border-r border-line bg-surface md:sticky md:top-0 md:flex md:h-screen md:w-64 md:shrink-0`}>
        <div className="hidden px-5 pb-2 pt-6 md:block">
          <Logo />
        </div>

        {memberships.length > 0 && activeOrg && (
          <div className="px-4 pt-4">
            <label htmlFor="org-switcher" className="mb-1 block text-xs font-medium text-muted">Organización</label>
            {memberships.length > 1 ? (
              <select id="org-switcher" value={activeOrg.id} disabled={switching}
                      onChange={(e) => onSwitch(e.target.value)}
                      className="w-full rounded-lg border border-line bg-white px-2.5 py-2 text-sm font-medium focus:border-accent focus:outline-none">
                {memberships.map((m) => (
                  <option key={m.organization.id} value={m.organization.id}>{m.organization.name}</option>
                ))}
              </select>
            ) : (
              <p className="rounded-lg border border-line bg-white px-2.5 py-2 text-sm font-medium">{activeOrg.name}</p>
            )}
            {activeRole && <p className="mt-1 text-xs text-muted">{ROLE_LABELS[activeRole]}</p>}
          </div>
        )}

        <nav className="mt-4 flex-1 space-y-0.5 px-3" aria-label="Principal">
          {links.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link key={item.href} href={item.href} onClick={() => setOpen(false)}
                    aria-current={active ? "page" : undefined}
                    className={`flex min-h-11 items-center gap-3 rounded-lg px-3 text-sm transition-colors ${active
                      ? "bg-accent-soft font-bold text-accent-strong"
                      : "text-slate-700 hover:bg-background hover:text-foreground"}`}>
                <svg aria-hidden="true" viewBox="0 0 24 24" className="size-[18px] shrink-0" fill="none" stroke="currentColor"
                     strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round"><path d={ICONS[item.href]} /></svg>
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="px-5 pb-3">
          <IndicadorConexion texto="Fuente oficial: CIMA · AEMPS" />
        </div>
        <div className="border-t border-line px-4 py-4">
          {IS_PREVIEW && <p className="mb-2 rounded-md bg-warning-soft px-2 py-1 text-xs text-amber-900">Modo vista previa · datos de ejemplo</p>}
          <p className="truncate text-sm font-medium">{user.displayName ?? user.email}</p>
          {user.displayName && <p className="truncate text-xs text-muted">{user.email}</p>}
          <button onClick={onSignOut} className="mt-2 text-xs text-muted underline-offset-2 hover:text-foreground hover:underline">
            Cerrar sesión
          </button>
        </div>
      </aside>
    </>
  );
}
