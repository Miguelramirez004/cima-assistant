"use client";

// Animaciones de la interfaz. Todas respetan "reducir movimiento" del sistema:
// las de CSS se detienen en globals.css y las que avanzan con temporizador
// se quedan en su estado final (usePrefersReducedMotion).

import { Children, useEffect, useState, useSyncExternalStore, type CSSProperties, type ReactNode } from "react";

const REDUCED_QUERY = "(prefers-reduced-motion: reduce)";

function subscribeReducedMotion(onChange: () => void) {
  const query = window.matchMedia(REDUCED_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

export function usePrefersReducedMotion() {
  return useSyncExternalStore(
    subscribeReducedMotion,
    () => window.matchMedia(REDUCED_QUERY).matches,
    () => false,
  );
}

/** Entrada escalonada: cada hijo aparece con un ligero desplazamiento. */
export function Aparecer({ children, paso = 0.05, retraso = 0, className = "" }:
  { children: ReactNode; paso?: number; retraso?: number; className?: string }) {
  return (
    <div className={className}>
      {Children.map(children, (child, i) => (
        <div className="cima-entra" style={{ animationDelay: `${retraso + i * paso}s` }}>{child}</div>
      ))}
    </div>
  );
}

/** Ficha técnica en miniatura recorrida por una línea de lectura. */
export function EscanerDocumento({ className = "" }: { className?: string }) {
  const lineas = [60, 100, 85, 100, 70, 100, 55];
  return (
    <div aria-hidden="true" className={`relative h-32 w-24 shrink-0 overflow-hidden rounded-lg border border-line bg-surface ${className}`}>
      <div className="absolute inset-x-3 top-3.5 flex flex-col gap-2">
        <span className="h-1.5 w-3/5 rounded-full bg-accent" />
        {lineas.map((w, i) => (
          <span key={i} className="cima-linea h-1 rounded-full bg-line"
                style={{ width: `${w}%`, animationDelay: `${i * 0.17}s` }} />
        ))}
      </div>
      <div className="cima-escaner absolute inset-x-0 h-0.5 bg-accent"
           style={{ boxShadow: "0 0 12px 3px color-mix(in srgb, var(--accent) 35%, transparent)" }} />
    </div>
  );
}

/** Pasos con indicador de carga, check dibujado y barra de progreso. */
export function PasosProgreso({ pasos, actual, barra = true }: { pasos: string[]; actual: number; barra?: boolean }) {
  const progreso = Math.min(100, Math.round(((actual + 0.6) / pasos.length) * 100));
  return (
    <div className="flex flex-col gap-4">
      <ol role="status" aria-live="polite" className="flex flex-col gap-3">
        {pasos.map((paso, i) => (
          <li key={`${i}-${paso}`} className={`flex items-center gap-3 text-sm ${i > actual ? "text-muted" : "text-foreground"} ${i === actual ? "font-semibold" : ""}`}>
            {i < actual && <Check className="size-5" />}
            {i === actual && <Spinner className="size-5" />}
            {i > actual && <span className="size-5 rounded-full border-2 border-dashed border-line" />}
            {paso}
          </li>
        ))}
      </ol>
      {barra && (
        <div aria-hidden="true" className="h-1 overflow-hidden rounded-full bg-surface">
          <div className="cima-barra h-full rounded-full bg-accent" style={{ width: `${progreso}%` }} />
        </div>
      )}
    </div>
  );
}

/** Tarjeta de trabajo en curso: ilustración animada junto a los pasos. */
export function ProcesoEnCurso({ ilustracion, pasos, actual, barra = true, nota }: {
  ilustracion: ReactNode; pasos: string[]; actual: number; barra?: boolean; nota?: string;
}) {
  return (
    <div className="cima-entra flex flex-wrap items-center gap-6 rounded-xl border border-line bg-white p-5">
      {ilustracion}
      <div className="min-w-60 flex-1">
        <PasosProgreso pasos={pasos} actual={actual} barra={barra} />
        {nota && <p className="mt-3 text-xs text-muted">{nota}</p>}
      </div>
    </div>
  );
}

export function Spinner({ className = "size-4", tono = "acento" }: { className?: string; tono?: "acento" | "claro" }) {
  const colores = tono === "claro" ? "border-white/40 border-t-white" : "border-accent-soft border-t-accent";
  return <span aria-hidden="true" className={`cima-gira inline-block shrink-0 rounded-full border-2 ${colores} ${className}`} />;
}

/** Check que se dibuja dentro de un círculo. */
export function Check({ className = "size-5" }: { className?: string }) {
  return (
    <span aria-hidden="true" className={`inline-flex shrink-0 items-center justify-center rounded-full bg-accent text-white ${className}`}>
      <svg viewBox="0 0 24 24" className="size-3/5" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
        <path className="cima-traza" pathLength={1} d="M20 6 9 17l-5-5" />
      </svg>
    </span>
  );
}

/** Éxito destacado: círculo y check que se trazan (guardado, copiado, invitación aceptada…). */
export function ExitoAnimado({ mensaje, className = "" }: { mensaje: string; className?: string }) {
  return (
    <div role="status" className={`flex flex-col items-center gap-3 text-center ${className}`}>
      <svg viewBox="0 0 52 52" className="size-14 text-accent" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle className="cima-traza" pathLength={1} cx="26" cy="26" r="23" style={{ animationDuration: "0.6s" }} />
        <path className="cima-traza" pathLength={1} d="m16 27 7 7 13-15" style={{ animationDelay: "0.45s" }} />
      </svg>
      <p className="cima-entra text-sm font-semibold" style={{ animationDelay: "0.6s" }}>{mensaje}</p>
    </div>
  );
}

/** Líneas provisionales con brillo mientras llega el contenido. */
export function Esqueleto({ lineas = 3, className = "" }: { lineas?: number; className?: string }) {
  const anchos = ["92%", "100%", "76%", "88%", "64%"];
  return (
    <div aria-hidden="true" className={`flex flex-col gap-2.5 ${className}`}>
      {Array.from({ length: lineas }, (_, i) => (
        <span key={i} className="cima-esqueleto h-3 rounded-full" style={{ width: anchos[i % anchos.length] }} />
      ))}
    </div>
  );
}

/** Cursor al final del texto que se está recibiendo en streaming. */
export function CursorEscritura() {
  return <span aria-hidden="true" className="cima-cursor ml-0.5 inline-block h-[1.1em] w-0.5 translate-y-[0.15em] bg-accent" />;
}

/** Formulación magistral: vaso de precipitados que se llena con burbujas. */
export function VasoFormulacion({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`relative h-32 w-24 shrink-0 ${className}`}>
      <div className="absolute inset-x-2 bottom-0 top-4 overflow-hidden rounded-b-2xl border-2 border-t-0 border-accent/60 bg-white">
        <div className="cima-llena absolute inset-0 bg-accent/20">
          <svg viewBox="0 0 200 20" preserveAspectRatio="none" className="cima-ola absolute -top-2.5 left-0 h-3 w-[200%] text-accent/20">
            <path fill="currentColor" d="M0 10 Q25 0 50 10 T100 10 T150 10 T200 10 V20 H0 Z" />
          </svg>
          {[18, 46, 70].map((left, i) => (
            <span key={left} className="cima-burbuja absolute bottom-2 size-2 rounded-full border border-accent/60 bg-white"
                  style={{ left: `${left}%`, animationDelay: `${i * 0.7}s` }} />
          ))}
        </div>
        {[30, 50, 70].map((top) => <span key={top} className="absolute right-0 h-0.5 w-3 bg-accent/40" style={{ top: `${top}%` }} />)}
      </div>
      <span className="absolute inset-x-0 top-3 h-1.5 rounded-full bg-accent/60" />
    </div>
  );
}

/** Prospectos: hoja que se despliega por paneles. */
export function ProspectoDesplegable({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`flex h-32 w-40 shrink-0 items-stretch ${className}`}>
      {[0, 1, 2].map((i) => (
        <div key={i} className="cima-despliega flex flex-1 flex-col gap-1.5 border border-line bg-surface p-2 first:rounded-l-lg last:rounded-r-lg"
             style={{ animationDelay: `${i * 0.25}s` }}>
          {i === 0 && <span className="h-1.5 w-4/5 rounded-full bg-accent" />}
          {[100, 80, 100, 60, 90].map((w, j) => <span key={j} className="h-1 rounded-full bg-line" style={{ width: `${w}%` }} />)}
        </div>
      ))}
    </div>
  );
}

/** Estado de conexión con latido suave. */
export function IndicadorConexion({ texto = "Conectado a CIMA · AEMPS" }: { texto?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted">
      <span className="cima-latido size-2 rounded-full bg-accent" />
      {texto}
    </span>
  );
}

/** Aviso flotante que entra desde abajo. */
export function Aviso({ children, tono = "exito" }: { children: ReactNode; tono?: "exito" | "error" }) {
  return (
    <div role="status" className={`cima-aviso inline-flex items-center gap-3 rounded-xl border px-4 py-3 text-sm font-medium shadow-lg ${tono === "exito" ? "border-line bg-white" : "border-danger/30 bg-danger-soft text-danger"}`}>
      {tono === "exito" ? <Check className="size-5" /> : <span aria-hidden="true" className="size-2 rounded-full bg-danger" />}
      {children}
    </div>
  );
}

/** Estructura molecular decorativa que se traza al cargar y queda fija. */
export function EstructuraMolecular({ className = "", style }: { className?: string; style?: CSSProperties }) {
  const trazos = [
    "M60 60 L94.6 80 L94.6 120 L60 140 L25.4 120 L25.4 80 Z",
    "M94.6 80 L129.2 60 L163.8 80 L163.8 120 L129.2 140 L94.6 120",
    "M94.6 120 L129.2 140 L129.2 180 L94.6 200 L60 180 L60 140",
    "M33 86 L33 114 M137 68 L156 79 M68 175 L68 147",
    "M60 60 L60 30 L84 16 M163.8 80 L196 62 M94.6 200 L94.6 228",
  ];
  return (
    <svg aria-hidden="true" viewBox="0 0 240 240" fill="none" stroke="currentColor" strokeWidth={1.4}
         strokeLinecap="round" strokeLinejoin="round" className={`pointer-events-none text-accent ${className}`} style={style}>
      {trazos.map((d, i) => <path key={i} className="cima-dibuja" pathLength={1} d={d} style={{ animationDelay: `${i * 0.3}s` }} />)}
    </svg>
  );
}

/** Texto de ejemplo que va rotando (para el placeholder de un campo). */
export function usePlaceholderRotativo(textos: string[], intervalo = 3200) {
  const reduced = usePrefersReducedMotion();
  const [i, setI] = useState(0);
  useEffect(() => {
    if (reduced || textos.length < 2) return;
    const id = setInterval(() => setI((n) => (n + 1) % textos.length), intervalo);
    return () => clearInterval(id);
  }, [reduced, textos.length, intervalo]);
  return textos[i];
}

/** Avanza un índice de 0 a total-1 y se detiene; al cambiar `clave` vuelve a empezar. */
export function useSecuencia(total: number, intervalo = 1000, clave: unknown = 0) {
  const reduced = usePrefersReducedMotion();
  const [estado, setEstado] = useState({ clave, paso: 0 });
  if (estado.clave !== clave) setEstado({ clave, paso: 0 });
  useEffect(() => {
    if (reduced) return;
    const id = setInterval(() => setEstado((e) => (e.paso < total - 1 ? { ...e, paso: e.paso + 1 } : e)), intervalo);
    return () => clearInterval(id);
  }, [total, intervalo, reduced, clave]);
  return reduced ? total - 1 : estado.paso;
}
