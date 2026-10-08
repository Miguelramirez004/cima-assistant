import { AuthCard } from "@/components/AuthCard";
import { InviteAccept } from "@/components/InviteAccept";

export const metadata = { title: "Invitación · CIMA Assistant" };

export default async function InvitePage({ params }: PageProps<"/invite/[token]">) {
  const { token } = await params;
  return (
    <AuthCard title="Invitación a CIMA Assistant"
              subtitle="Le han invitado a unirse a una organización para consultar información oficial de medicamentos.">
      <InviteAccept token={token} />
    </AuthCard>
  );
}
