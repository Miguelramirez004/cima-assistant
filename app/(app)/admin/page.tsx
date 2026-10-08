import { notFound } from "next/navigation";

import { AdminPanel } from "@/components/AdminPanel";
import { PageHeader } from "@/components/ui";
import { getSessionContext, listAllOrganizations } from "@/lib/data";

export const metadata = { title: "Administración · CIMA Assistant" };

export default async function AdminPage() {
  const { user } = await getSessionContext();
  if (!user.isPlatformAdmin) notFound();
  return (
    <>
      <PageHeader title="Administración" description="Alta de organizaciones clientes y de sus propietarios." />
      <AdminPanel organizations={await listAllOrganizations()} />
    </>
  );
}
