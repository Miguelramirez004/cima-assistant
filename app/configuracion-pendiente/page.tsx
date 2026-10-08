import { AuthCard } from "@/components/AuthCard";

export const metadata = { title: "Configuración pendiente · CIMA Assistant" };

/** Se muestra mientras el despliegue no tenga las variables de Supabase. */
export default function ConfiguracionPendientePage() {
  return (
    <AuthCard title="Configuración pendiente"
              subtitle="Esta instalación de CIMA Assistant aún no está conectada a Supabase.">
      <div className="space-y-3 text-sm">
        <p>Añada estas variables de entorno en Vercel (Settings → Environment Variables) y vuelva a desplegar:</p>
        <ul className="list-disc space-y-1 pl-5 font-mono text-xs">
          <li>NEXT_PUBLIC_SUPABASE_URL</li>
          <li>NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY</li>
          <li>SUPABASE_SECRET_KEY</li>
          <li>APP_URL</li>
        </ul>
        <p className="text-muted">Consulte docs/SUPABASE_SETUP.md en el repositorio.</p>
      </div>
    </AuthCard>
  );
}
