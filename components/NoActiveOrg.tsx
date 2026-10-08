import Link from "next/link";

import { EmptyState } from "@/components/ui";

/** Para administradores de plataforma que aún no pertenecen a ninguna organización. */
export function NoActiveOrg() {
  return (
    <EmptyState title="No pertenece a ninguna organización">
      Cree una organización desde <Link href="/admin" className="text-accent-strong underline">Administración</Link> o
      acepte una invitación para usar esta sección.
    </EmptyState>
  );
}
