import { AuthCard } from "@/components/AuthCard";
import { SignOutButton } from "@/components/SignOutButton";

export const metadata = { title: "Sin organización · CIMA Assistant" };

export default function SinOrganizacionPage() {
  return (
    <AuthCard title="Aún no pertenece a ninguna organización"
              subtitle="Para usar CIMA Assistant necesita una invitación del responsable de su organización. Abra el enlace del email de invitación o pídale que le invite de nuevo.">
      <SignOutButton />
    </AuthCard>
  );
}
