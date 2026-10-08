// Página provisional (fase 1). Las secciones Formulación, Consultas,
// Prospectos e Historial, con login de Supabase, llegan en la fase 5.
export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center gap-6 px-4 py-24">
      <p className="text-sm font-semibold uppercase tracking-wide text-accent">
        CIMA Assistant
      </p>
      <h1 className="text-3xl font-semibold tracking-tight">
        Migración a Vercel + Supabase en curso
      </h1>
      <p className="text-muted">
        Formulación magistral, consultas y prospectos a partir de la información
        oficial de CIMA (AEMPS). Esta versión aún no está operativa; la
        aplicación actual sigue disponible en Streamlit.
      </p>
      <a
        href="/api/health"
        className="w-fit rounded-lg border border-line px-4 py-2 text-sm hover:border-accent"
      >
        Estado de la API
      </a>
    </main>
  );
}
