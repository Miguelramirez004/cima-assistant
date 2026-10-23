"use client";

import { useState, type ReactNode } from "react";

import {
  Aparecer, Aviso, CursorEscritura, Esqueleto, EscanerDocumento, EstructuraMolecular, ExitoAnimado,
  IndicadorConexion, PasosProgreso, ProspectoDesplegable, Spinner, VasoFormulacion,
  usePlaceholderRotativo, useSecuencia,
} from "@/components/motion";
import { Button, inputClass } from "@/components/ui";

const PASOS = ["Buscando medicamentos en CIMA", "Leyendo las fichas técnicas", "Redactando la respuesta con citas"];

function Muestra({ titulo, uso, children }: { titulo: string; uso: string; children: ReactNode }) {
  return (
    <section className="cima-elevable flex flex-col gap-4 rounded-xl border border-line bg-white p-5">
      <div className="flex min-h-36 items-center justify-center rounded-lg bg-surface p-4">{children}</div>
      <div>
        <h2 className="text-sm font-semibold">{titulo}</h2>
        <p className="mt-1 text-xs text-muted">{uso}</p>
      </div>
    </section>
  );
}

export function MotionShowcase() {
  const [ronda, setRonda] = useState(0);
  const paso = useSecuencia(PASOS.length + 1, 1100, ronda);
  const placeholder = usePlaceholderRotativo([
    "Escriba un medicamento o una pregunta…",
    "¿Dosis de amoxicilina en niños?",
    "¿Interacciones de la warfarina?",
  ]);

  return (
    <main className="relative mx-auto w-full max-w-6xl px-4 py-10">
      <EstructuraMolecular className="absolute -top-2 right-4 hidden size-48 opacity-20 md:block" key={`mol-${ronda}`} />
      <header className="relative mb-8 flex flex-col items-start gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-accent">CIMA Assistant</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Animaciones de la interfaz</h1>
          <p className="mt-1 max-w-xl text-sm text-muted">Componentes de components/motion.tsx. Todos respetan la opción del sistema «reducir movimiento».</p>
        </div>
        <Button variant="secondary" onClick={() => setRonda((r) => r + 1)}>Repetir animaciones</Button>
      </header>

      <Aparecer key={ronda} className="relative grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-4">
        <Muestra titulo="Lectura de ficha técnica" uso="Consultas: mientras se leen las fichas en CIMA.">
          <EscanerDocumento />
        </Muestra>
        <Muestra titulo="Pasos con progreso" uso="Consultas, formulación y prospectos: eventos de traza del backend.">
          <div className="w-full"><PasosProgreso pasos={PASOS} actual={Math.min(paso, PASOS.length)} /></div>
        </Muestra>
        <Muestra titulo="Preparando formulación" uso="Formulación magistral: mientras se genera la fórmula.">
          <VasoFormulacion />
        </Muestra>
        <Muestra titulo="Desplegando prospecto" uso="Prospectos: mientras se redacta el prospecto para el paciente.">
          <ProspectoDesplegable />
        </Muestra>
        <Muestra titulo="Contenido cargando" uso="Historial, administración y cualquier lista que espera datos.">
          <Esqueleto lineas={4} className="w-full" />
        </Muestra>
        <Muestra titulo="Respuesta en streaming" uso="Cursor al final del texto que va llegando.">
          <p className="text-sm leading-relaxed">Según la ficha técnica, omeprazol puede utilizarse durante el embarazo<CursorEscritura /></p>
        </Muestra>
        <Muestra titulo="Acción completada" uso="Guardar organización, aceptar invitación, copiar o descargar.">
          <ExitoAnimado key={`ok-${ronda}`} mensaje="Formulación guardada" />
        </Muestra>
        <Muestra titulo="Avisos" uso="Confirmaciones y errores breves, en la esquina de la pantalla.">
          <div className="flex flex-col items-center gap-2">
            <Aviso key={`a-${ronda}`}>Enlace de invitación copiado</Aviso>
            <Aviso key={`b-${ronda}`} tono="error">No se pudo conectar con CIMA</Aviso>
          </div>
        </Muestra>
        <Muestra titulo="Ejemplos rotativos" uso="Placeholder del campo de consulta.">
          <label className="w-full">
            <span className="sr-only">Consulta</span>
            <input key={placeholder} className={`${inputClass} cima-funde`} placeholder={placeholder} />
          </label>
        </Muestra>
        <Muestra titulo="Botón ocupado" uso="Cualquier envío: el botón muestra que está trabajando.">
          <Button disabled><Spinner tono="claro" />Consultando…</Button>
        </Muestra>
        <Muestra titulo="Estado de conexión" uso="Pie del menú lateral.">
          <IndicadorConexion />
        </Muestra>
        <Muestra titulo="Tarjeta con elevación" uso="Sugerencias y tarjetas clicables: clase cima-elevable.">
          <button type="button" className="cima-elevable w-full rounded-xl border border-line bg-white p-4 text-left text-sm font-medium">
            ¿Puedo tomar omeprazol durante el embarazo?
          </button>
        </Muestra>
      </Aparecer>
    </main>
  );
}
