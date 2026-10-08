"use client";

import { useState } from "react";

import { useActiveOrg } from "@/components/SessionProvider";
import { downloadText, slugify } from "@/components/download";
import { Alert, Button, Card, Spinner, inputClass } from "@/components/ui";
import { generateProspecto } from "@/lib/client-actions";
import type { ProspectoResult } from "@/lib/types";

const EXAMPLES = [
  "Prospecto de ibuprofeno 600 mg comprimidos",
  "Prospecto de amoxicilina 500 mg cápsulas",
  "Prospecto de omeprazol 20 mg",
];

export function ProspectoView({ initialQuery = "" }: { initialQuery?: string }) {
  const { org } = useActiveOrg();
  const [query, setQuery] = useState(initialQuery);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ query: string; data: ProspectoResult } | null>(null);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const text = query.trim();
    if (text.length < 3) return;
    setLoading(true);
    setError(null);
    try {
      setResult({ query: text, data: await generateProspecto(org.id, text) });
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo generar el prospecto");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label htmlFor="query" className="mb-1.5 block text-sm font-medium">Medicamento</label>
            <input id="query" value={query} maxLength={2000} required minLength={3}
                   onChange={(e) => setQuery(e.target.value)}
                   placeholder="Ej.: Prospecto de ibuprofeno 600 mg comprimidos" className={inputClass} />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((example) => (
                <button key={example} type="button" onClick={() => setQuery(example)}
                        className="rounded-full border border-line px-3 py-1 text-xs text-slate-600 hover:border-accent hover:text-accent-strong">
                  {example}
                </button>
              ))}
            </div>
            <Button type="submit" disabled={loading || query.trim().length < 3}>
              {loading ? "Generando…" : "Generar prospecto"}
            </Button>
          </div>
        </form>
      </Card>

      {loading && <Spinner label="Obteniendo el prospecto registrado en CIMA…" />}
      {error && <Alert>{error}</Alert>}

      {result && (
        <Card>
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-muted">Prospecto</p>
              <p className="mt-1 font-medium">{result.data.medication_name ?? result.query}</p>
              {result.data.nregistro && (
                <a href={`https://cima.aemps.es/cima/dochtml/p/${result.data.nregistro}/Prospecto.html`}
                   target="_blank" rel="noopener noreferrer" className="text-xs text-accent-strong underline">
                  Nº Registro {result.data.nregistro} · prospecto oficial en CIMA
                </a>
              )}
            </div>
            {result.data.success && (
              <Button variant="secondary"
                      onClick={() => downloadText(`prospecto-${slugify(result.data.medication_name ?? result.query)}.txt`,
                                                  result.data.content, "text/plain;charset=utf-8")}>
                Descargar .txt
              </Button>
            )}
          </div>
          {result.data.success
            ? <div className="whitespace-pre-wrap text-sm leading-relaxed">{result.data.content}</div>
            : <Alert>{result.data.content}</Alert>}
        </Card>
      )}
    </div>
  );
}
