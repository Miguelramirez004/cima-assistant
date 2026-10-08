"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { useSession } from "@/components/SessionProvider";
import { signOut, switchOrganization } from "@/lib/client-actions";
import { IS_PREVIEW } from "@/lib/env";
import { ROLE_LABELS } from "@/lib/types";

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
      <div className="flex items-center justify-between border-b border-line bg-white px-4 py-3 md:hidden">
        <span className="font-semibold">CIMA Assistant</span>
        <button onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="sidebar"
                className="rounded-md border border-line px-3 py-1 text-sm">
          {open ? "Cerrar" : "Menú"}
        </button>
      </div>

      <aside id="sidebar"
             className={`${open ? "flex" : "hidden"} w-full flex-col border-r border-line bg-surface md:sticky md:top-0 md:flex md:h-screen md:w-64 md:shrink-0`}>
        <div className="hidden px-5 pb-2 pt-6 md:block">
          <p className="text-xs font-semibold uppercase tracking-wider text-accent">CIMA Assistant</p>
          <p className="mt-0.5 text-xs text-muted">Información oficial AEMPS</p>
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
                    className={`block rounded-lg px-3 py-2 text-sm ${active
                      ? "bg-white font-medium text-accent-strong shadow-sm ring-1 ring-line"
                      : "text-slate-600 hover:bg-white hover:text-foreground"}`}>
                {item.label}
              </Link>
            );
          })}
        </nav>

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
