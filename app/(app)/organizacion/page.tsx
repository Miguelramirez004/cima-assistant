import { NoActiveOrg } from "@/components/NoActiveOrg";
import { OrganizationAdmin } from "@/components/OrganizationAdmin";
import { UsagePanel } from "@/components/UsagePanel";
import { PageHeader } from "@/components/ui";
import { getSessionContext, getUsageSummary, listMembers, listPendingInvitations } from "@/lib/data";

export const metadata = { title: "Organización · CIMA Assistant" };

export default async function OrganizacionPage() {
  const { activeOrg, activeRole } = await getSessionContext();
  if (!activeOrg || !activeRole) return <NoActiveOrg />;
  const canManage = activeRole === "owner" || activeRole === "admin";

  const members = await listMembers(activeOrg.id);
  const [invitations, usage] = await Promise.all([
    canManage ? listPendingInvitations(activeOrg.id) : Promise.resolve([]),
    getUsageSummary(activeOrg, members),
  ]);

  return (
    <>
      <PageHeader title={activeOrg.name} description="Miembros, invitaciones y consumo de la organización." />
      <div className="space-y-6">
        <UsagePanel usage={usage} orgWide={canManage} />
        <OrganizationAdmin key={activeOrg.id} members={members} invitations={invitations} />
      </div>
    </>
  );
}
