// Configuración pública del frontend.

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_PUBLISHABLE_KEY =
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
  "";

// Modo vista previa: datos de ejemplo sin Supabase ni OpenAI, para trabajar
// la interfaz en local. Nunca se activa en producción.
export const IS_PREVIEW =
  process.env.NEXT_PUBLIC_PREVIEW_MODE === "1" && process.env.NODE_ENV !== "production";

/** Sin URL ni clave pública de Supabase la app no puede autenticar a nadie. */
export const SUPABASE_CONFIGURED = Boolean(SUPABASE_URL && SUPABASE_PUBLISHABLE_KEY);
