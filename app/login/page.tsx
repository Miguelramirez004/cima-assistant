import { AuthCard } from "@/components/AuthCard";
import { LoginForm } from "@/components/LoginForm";
import { safeNext } from "@/lib/url";

export const metadata = { title: "Iniciar sesión · CIMA Assistant" };

const ERRORS: Record<string, string> = {
  link: "El enlace de acceso no es válido o ha caducado. Solicite uno nuevo.",
};

export default async function LoginPage({ searchParams }: PageProps<"/login">) {
  const { next, error } = await searchParams;
  return (
    <AuthCard title="Iniciar sesión" subtitle="Le enviaremos un enlace de acceso a su email. El acceso es solo por invitación.">
      <LoginForm next={safeNext(next)} initialError={typeof error === "string" ? ERRORS[error] : undefined} />
    </AuthCard>
  );
}
