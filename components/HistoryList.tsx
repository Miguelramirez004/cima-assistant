"use client";

import Link from "next/link";
import { useState } from "react";

import { Markdown } from "@/components/Markdown";
import { References } from "@/components/References";
import { downloadText, slugify } from "@/components/download";
import { Alert, Button, EmptyState, formatDate } from "@/components/ui";
import { deleteHistoryItem } from "@/lib/client-actions";
import type { Conversation, Formulation, Prospecto } from "@/lib/types";

type Item =
  | { kind: "formulations"; data: Formulation }
  | { kind: "prospectos"; data: Prospecto }
  | { kind: "conversations"; data: Conversation };

const EMPTY: Record<Item["kind"], { title: string; href: string; cta: string }> = {
  formulations: { title: "Aún no hay formulaciones", href: "/formulacion", cta: "Generar una formulación" },
  prospectos: { title: "Aún no hay prospectos", href: "/prospectos", cta: "Generar un prospecto" },
  conversations: { title: "Aún no hay conversaciones", href: "/consultas", cta: "Hacer una consulta" },
};

export function HistoryList({ kind, items: initialItems }: { kind: Item["kind"]; items: Item["data"][] }) {
  const [items, setItems] = useState(initialItems);
  const [error, setError] = useState<string | null>(null);

  const remove = async (id: string) => {
    if (!confirm("¿Eliminar este elemento del historial? No se puede deshacer.")) return;
    setError(null);
    try {
      await deleteHistoryItem(kind, id);
      setItems((list) => list.filter((item) => item.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo eliminar");
    }
  };

  if (!items.length) {
    const empty = EMPTY[kind];
    return (
      <EmptyState title={empty.title}>
        <Link href={empty.href} className="text-accent-strong underline">{empty.cta}</Link>
      </EmptyState>
    );
  }

  return (
    <div className="space-y-3">
      {error && <Alert>{error}</Alert>}
      <ul className="divide-y divide-line rounded-xl border border-line bg-white">
        {items.map((data) => (
          <li key={data.id} className="px-5 py-4">
            {kind === "conversations"
              ? <ConversationRow conversation={data as Conversation} onDelete={() => remove(data.id)} />
              : <DocumentRow kind={kind} data={data as Formulation | Prospecto} onDelete={() => remove(data.id)} />}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ConversationRow({ conversation, onDelete }: { conversation: Conversation; onDelete: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="min-w-0">
        <Link href={`/consultas/${conversation.id}`} className="block truncate font-medium hover:text-accent-strong">
          {conversation.title ?? "Sin título"}
        </Link>
        <p className="text-xs text-muted">Última actividad: {formatDate(conversation.updated_at)}</p>
      </div>
      <Button variant="ghost" onClick={onDelete} aria-label="Eliminar conversación">Eliminar</Button>
    </div>
  );
}

function DocumentRow({ kind, data, onDelete }: {
  kind: "formulations" | "prospectos"; data: Formulation | Prospecto; onDelete: () => void;
}) {
  const isFormulation = kind === "formulations";
  const body = isFormulation ? (data as Formulation).answer : (data as Prospecto).content;
  const title = isFormulation ? data.query : (data as Prospecto).medication_name ?? data.query;

  const download = () => isFormulation
    ? downloadText(`formulacion-${slugify(data.query)}.md`, `# Formulación magistral\n\n## Consulta\n${data.query}\n\n${body}\n`)
    : downloadText(`prospecto-${slugify(title)}.txt`, body, "text/plain;charset=utf-8");

  return (
    <details className="group">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-medium group-open:whitespace-normal">{title}</p>
          <p className="text-xs text-muted">{formatDate(data.created_at)}</p>
        </div>
        <span className="shrink-0 text-xs text-muted group-open:hidden">Ver</span>
      </summary>
      <div className="mt-4 border-t border-line pt-4">
        {isFormulation
          ? <><Markdown>{body}</Markdown><References references={(data as Formulation).references ?? []} /></>
          : <div className="whitespace-pre-wrap text-sm leading-relaxed">{body}</div>}
        <div className="mt-4 flex gap-2">
          <Button variant="secondary" onClick={download}>Descargar</Button>
          <Button variant="danger" onClick={onDelete}>Eliminar</Button>
        </div>
      </div>
    </details>
  );
}
