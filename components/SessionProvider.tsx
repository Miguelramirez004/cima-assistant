"use client";

import { createContext, useContext, type ReactNode } from "react";

import type { SessionContext } from "@/lib/types";

const Context = createContext<SessionContext | null>(null);

export function SessionProvider({ value, children }: { value: SessionContext; children: ReactNode }) {
  return <Context value={value}>{children}</Context>;
}

export function useSession(): SessionContext {
  const value = useContext(Context);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}

/** Organización activa; las páginas de generación solo se muestran si existe. */
export function useActiveOrg() {
  const { activeOrg, activeRole } = useSession();
  if (!activeOrg || !activeRole) throw new Error("No active organization");
  return { org: activeOrg, role: activeRole };
}
