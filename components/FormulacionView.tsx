"use client";

import Link from "next/link";
import { useState } from "react";

import { Markdown } from "@/components/Markdown";
import { References } from "@/components/References";
import { useActiveOrg } from "@/components/SessionProvider";
import { downloadText, slugify } from "@/components/download";
import { Alert, Button, Card, Spinner, inputClass } from "@/components/ui";
import { generateFormulacion } from "@/lib/client-actions";
import type { FormulacionResult } from "@/lib/types";

const EXAMPLES = [
  "Suspensión oral de omeprazol 2 mg/ml para uso pediátrico",
  "Gel de metronidazol 0,75 % para uso tópico",
  "Solución oral de propranolol 1 mg/ml",
  "Cápsulas de melatonina 3 mg",
];

export function FormulacionView() {
  const { org } = useActiveOrg();
  const [query, setQuery] = useState("");
  const [advanced, setAdvanced] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ query: string; data: FormulacionResult } | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const text = query.trim();
    if (text.length < 3) return;
    setLoading(true);
    setError(null);
    try {
      setResult({ query: text, data: await generateFormulacion(org.id, text, advanced) });
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo generar la formulación");
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    if (!result) return;
    const { query: q, data } = result;
    downloadText(`formulacion-${slugify(q)}.md`,
      `# Formulación magistral\n\n## Consulta\n${q}\n\n## Formulación\n${data.answer}\n\n## Contexto CIMA\n${data.context}\n`);
  };

  return (
    <div className="space-y-6">
      <Card>
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label htmlFor="query" className="mb-1.5 block text-sm font-medium">
              Principio activo, concentración y forma farmacéutica
            </label>
            <textarea id="query" rows={3} maxLength={2000} value={query} required minLength={3}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e); }}
                      placeholder="Ej.: Suspensión oral de omeprazol 2 mg/ml para uso pediátrico"
                      className={inputClass} />
          </div>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button key={example} type="button" onClick={() => setQuery(example)}
                      className="rounded-full border border-line px-3 py-1 text-xs text-slate-600 hover:border-accent hover:text-accent-strong">
                {example}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)}
                     className="h-4 w-4 accent-[var(--accent)]" />
              Búsqueda avanzada en CIMA
            </label>
            <Button type="submit" disabled={loading || query.trim().length < 3}>
              {loading ? "Generando…" : "Generar formulación"}
            </Button>
          </div>
        </form>
      </Card>

      {loading && <Spinner label="Consultando CIMA y redactando la formulación (puede tardar hasta un minuto)…" />}
      {error && <Alert>{error}</Alert>}

      {result?.data.redirect === "prospecto" && (
        <Alert tone="info">
          Esta consulta pide un prospecto.{" "}
          <Link href={`/prospectos?q=${encodeURIComponent(result.query)}`} className="font-medium underline">
            Generarlo en Prospectos
          </Link>
        </Alert>
      )}

      {result && !result.data.redirect && (
        <Card>
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-muted">Formulación</p>
              <p className="mt-1 font-medium">{result.query}</p>
            </div>
            <Button variant="secondary" onClick={download}>Descargar .md</Button>
          </div>
          {result.data.success ? <Markdown>{result.data.answer}</Markdown> : <Alert>{result.data.answer}</Alert>}
          <References references={result.data.references} />
          {result.data.context && (
            <details className="mt-5 rounded-lg border border-line bg-surface">
              <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-slate-600">Ver contexto de CIMA utilizado</summary>
              <pre className="max-h-96 overflow-auto whitespace-pre-wrap px-4 pb-4 text-xs text-slate-600">{result.data.context}</pre>
            </details>
          )}
          <p className="mt-5 text-xs text-muted">
            Revise siempre la formulación con un farmacéutico cualificado antes de elaborarla.
          </p>
        </Card>
      )}
    </div>
  );
}
